"""
Agent 2: Regulatory Entitlement & Self-Verification Agent, built with LangGraph (A1, A3, F4-F7).

Graph:

    START -> plan --(tool chosen)--> run_tool -> plan -> ...
               \\--(finish)--> decide -> verify --(pass)--> finalize -> END
                                           |--(retry: re-search with another strategy)--> plan
                                           \\--(still unsupported)--> escalate -> END

* plan      The agent chooses the next tool call. With a real LLM configured, the
            model picks the tool and its arguments from the claim, the clauses read
            so far and the unread cross-references. In stub mode (load tests) a
            deterministic policy makes the same kind of choices.
* run_tool  Executes corpus_search / lookup_clause_by_id / get_claim_field, adds the
            passages to the evidence, and scans them for references to other
            provisions ("under Clause 3.7 and 3.8", "Provisions of 3.3.2 shall
            apply"). Unread references become pending, so the agent goes and reads
            the provision that modifies the one it found (spec section 5, steps 1-3).
* decide    Evidence-gated entitlement calculation (see entitlement_rules.py): only
            provisions present in the retrieved passages can be applied or cited.
* verify    Every citation must resolve to a line of retrieved text (A3). If not,
            the agent re-searches with a different strategy (direct clause lookup +
            hybrid search) up to MAX_VERIFY_RETRIES times, then escalates.
            An unverified answer is never returned.

Loop bounds (A1): max graph steps, max tool calls and a wall-clock deadline. Hitting
any bound means the run did not complete, so it is escalated, not answered (A5).
"""

import json
import re
import time
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, START, END

from src.agents import entitlement_rules as rules
from src.agents.llm_client import LLMClient
from src.core.contracts import Citation, RoutingDecision
from src.core.observability import (
    AGENT_STEP_LATENCY,
    RETRY_COUNTER,
    VERIFICATION_COUNTER,
    logger,
)
from src.core.tracing import span
from src.tools.claim_tools import ClaimToolBox

MAX_VERIFY_RETRIES = 2
MAX_PENDING_REFS = 6
TOOLS = ("corpus_search", "lookup_clause_by_id", "get_claim_field", "finish")

_REF_RE = re.compile(
    r"(?:clause|paragraph|provisions? of|as per|under|car)\s+"
    r"(\d+\.\d+(?:\.\d+)?(?:\s*(?:,|and|&|/|or)\s*\d+\.\d+(?:\.\d+)?)*)",
    re.IGNORECASE,
)
_NUM_RE = re.compile(r"\d+\.\d+(?:\.\d+)?")

# Facts a provision may hinge on: if a retrieved provision mentions the phrase,
# the agent checks the corresponding field of the claim with get_claim_field.
_FIELD_TRIGGERS = [
    ("contact details", "contact_details_provided_at_booking"),
    ("extraordinary circumstances", "cause_attribution"),
    ("accepts an alternate flight", "accepted_alternate_flight"),
    ("informed", "notice_hours_before_departure"),
]
_MODIFIER_TERMS = ["alternate flight", "contact details", "extraordinary circumstances", "refund",
                   "meals", "hotel", "block time", "notice", "volunteers"]


class AgentState(TypedDict, total=False):
    handoff: Dict[str, Any]
    passages: List[Dict[str, Any]]
    pending_refs: List[str]
    followed_refs: List[str]
    queries_done: List[str]
    modifier_search_done: bool
    fields_checked: List[str]
    next_action: Dict[str, Any]
    steps: int
    verify_attempts: int
    strategy: str
    decision: Dict[str, Any]
    verification: Dict[str, Any]
    result: Dict[str, Any]
    trace: List[Dict[str, Any]]
    stop_reason: Optional[str]


class RegulatoryReasoningAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: str,
        toolbox: ClaimToolBox,
        prompt_version: str = "v2.0",
        max_steps: int = 16,
        max_tool_calls: int = 10,
        deadline: Optional[float] = None,
        correlation_id: Optional[str] = None,
        demo_fault: Optional[str] = None,
    ):
        self.llm = llm_client
        self.system_prompt = system_prompt
        self.toolbox = toolbox
        self.baseline = prompt_version in ("v1", "v1.0")
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls
        self.deadline = deadline or (time.time() + 45.0)
        self.correlation_id = correlation_id
        self.demo_fault = demo_fault
        self.degraded_reasons: List[str] = []
        self.graph = self._build_graph()

    # ================================================================== graph
    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("plan", self._timed("plan", self._plan))
        g.add_node("run_tool", self._timed("run_tool", self._run_tool))
        g.add_node("decide", self._timed("decide", self._decide))
        g.add_node("verify", self._timed("verify", self._verify))
        g.add_node("escalate", self._timed("escalate", self._escalate))
        g.add_node("finalize", self._timed("finalize", self._finalize))

        g.add_edge(START, "plan")
        g.add_conditional_edges("plan", lambda s: "decide" if s["next_action"]["tool"] == "finish" else "run_tool",
                                {"decide": "decide", "run_tool": "run_tool"})
        g.add_edge("run_tool", "plan")
        g.add_edge("decide", "verify")
        g.add_conditional_edges("verify", lambda s: s["verification"]["route"],
                                {"pass": "finalize", "retry": "plan", "escalate": "escalate"})
        g.add_edge("escalate", END)
        g.add_edge("finalize", END)
        return g.compile()

    def _timed(self, name, fn):
        def wrapper(state: AgentState) -> AgentState:
            t0 = time.time()
            with span(f"agent2.{name}", {"correlation_id": self.correlation_id, "node": name}):
                update = fn(state)
            dt = time.time() - t0
            AGENT_STEP_LATENCY.labels(agent_name="reasoner", step=name).observe(dt)
            trace = list(state.get("trace", []))
            event = {"node": name, "latency_seconds": round(dt, 4)}
            event.update(update.pop("_event", {}))
            trace.append(event)
            update["trace"] = trace
            return update
        return wrapper

    # ================================================================== entry point
    def evaluate_entitlement(self, handoff: Dict[str, Any]) -> Dict[str, Any]:
        state: AgentState = {
            "handoff": handoff, "passages": [], "pending_refs": [], "followed_refs": [],
            "queries_done": [], "modifier_search_done": False, "fields_checked": [],
            "steps": 0, "verify_attempts": 0, "strategy": "dense", "trace": [], "stop_reason": None,
        }
        final = self.graph.invoke(state, config={"recursion_limit": 4 * self.max_steps + 20})
        return final["result"]

    # ================================================================== nodes
    def _limit_hit(self, state: AgentState) -> Optional[str]:
        if time.time() > self.deadline:
            return "wall-clock timeout"
        if state["steps"] >= self.max_steps:
            return f"step limit ({self.max_steps})"
        if self.toolbox.get_total_calls() >= self.max_tool_calls:
            return f"tool-call limit ({self.max_tool_calls})"
        return None

    def _plan(self, state: AgentState) -> AgentState:
        steps = state["steps"] + 1
        limit = self._limit_hit({**state, "steps": state["steps"]})
        if limit:
            action = {"tool": "finish", "args": {}, "reason": f"loop bound reached: {limit}", "by": "bound"}
            return {"steps": steps, "next_action": action, "stop_reason": limit,
                    "_event": {"action": action}}

        policy_action = self._policy_action(state)
        action = policy_action
        if not self.baseline and self.llm.is_stub:
            # In-process stub: make the call for latency/cost accounting, then follow the policy.
            try:
                self.llm.generate(self.system_prompt, f"plan step {steps}", step_name="agent_2_plan")
            except Exception as e:
                self._degrade(f"planner model unavailable ({type(e).__name__})")
        elif not self.baseline:
            llm_action = self._llm_action(state)
            if llm_action is not None:
                if llm_action["tool"] == "finish" and (state["pending_refs"] or not state["modifier_search_done"]):
                    why = ("unread cross-references remain" if state["pending_refs"]
                           else "no search for modifying conditions/exemptions has been run yet")
                    action = {**policy_action, "reason": policy_action["reason"] +
                              f" (model wanted to finish; {why})", "by": "policy-override"}
                else:
                    action = llm_action
        return {"steps": steps, "next_action": action, "_event": {"action": action}}

    def _policy_action(self, state: AgentState) -> Dict[str, Any]:
        """Deterministic planner: used in stub mode and as the fallback when the model's choice is unusable."""
        h = state["handoff"]
        dtype = str(getattr(h["disruption_type"], "value", h["disruption_type"]))
        read_text = " ".join(p["text"] for p in state["passages"]).lower()

        if not state["queries_done"]:
            query = f"{dtype.replace('_', ' ')} compensation entitlement. {h['free_text_claim_narrative'][:300]}"
            return {"tool": "corpus_search", "args": {"query": query, "strategy": state["strategy"]},
                    "reason": "retrieve the provision that appears to apply", "by": "policy"}

        if self.baseline:  # v1.0 baseline: single retrieval pass, answer from what came back
            return {"tool": "finish", "args": {}, "reason": "baseline variant answers after one retrieval", "by": "policy"}

        if state["pending_refs"]:
            ref = state["pending_refs"][0]
            return {"tool": "lookup_clause_by_id", "args": {"clause_id": ref},
                    "reason": f"a retrieved provision refers to {ref}; read it to see whether it modifies the answer",
                    "by": "policy"}

        cause = str(h.get("cause_attribution", "")).lower()
        if ("disputed" in cause or "contested" in cause) and not any(
                rules.clause_compatible("CAR-3.6", p["clause_id"]) for p in state["passages"]):
            return {"tool": "lookup_clause_by_id", "args": {"clause_id": "CAR-3.6"},
                    "reason": "cause is disputed: read the dispute-resolution provisions", "by": "policy"}

        if not state["modifier_search_done"]:
            found = [t for t in _MODIFIER_TERMS if t in read_text][:4]
            query = (f"airline shall not be liable conditions exemptions {dtype.replace('_', ' ')} "
                     + " ".join(found))
            return {"tool": "corpus_search", "args": {"query": query, "strategy": "hybrid"},
                    "reason": "search again with a query derived from what was read, for conditions and "
                              "exemptions that could remove or modify the entitlement", "by": "policy"}

        for phrase, field in _FIELD_TRIGGERS:
            if phrase in read_text and field not in state["fields_checked"] and len(state["fields_checked"]) < 2:
                return {"tool": "get_claim_field", "args": {"field_name": field},
                        "reason": f"a retrieved provision depends on '{phrase}'; check the claim's {field}",
                        "by": "policy"}

        return {"tool": "finish", "args": {}, "reason": "no unread references or unchecked conditions remain",
                "by": "policy"}

    def _degrade(self, reason: str):
        if reason not in self.degraded_reasons:
            self.degraded_reasons.append(reason)

    def _llm_action(self, state: AgentState) -> Optional[Dict[str, Any]]:
        """Ask the model to choose the next tool call (A1). Returns None if the reply is unusable."""
        h = state["handoff"]
        read = [f"- {p['clause_id']}: {p['text'][:160].replace(chr(10), ' ')}" for p in state["passages"]]
        prompt = (
            "CLAIM FACTS:\n" + json.dumps({k: (getattr(v, 'value', v)) for k, v in h.items()
                                           if k not in ("llm_classification_summary",)}, default=str)[:1500]
            + "\n\nCLAUSES READ SO FAR:\n" + ("\n".join(read) or "(none)")
            + f"\n\nUNREAD CROSS-REFERENCES FOUND IN THOSE CLAUSES: {state['pending_refs'] or 'none'}"
            + f"\nQUERIES ALREADY RUN: {state['queries_done'] or 'none'}"
            + f"\nTOOL CALLS LEFT: {self.max_tool_calls - self.toolbox.get_total_calls()}"
            + "\n\nChoose the NEXT action. Tools:\n"
              "- corpus_search(query, strategy in [dense, keyword, hybrid]): search the CAR and airline charter\n"
              "- lookup_clause_by_id(clause_id): read one clause and its sub-clauses, e.g. CAR-3.7\n"
              "- get_claim_field(field_name): read one field of the claim\n"
              "- finish(): stop when every provision that could create, remove or modify the entitlement "
              "(conditions, exemptions, referenced facilities clauses) has been read\n"
              'Reply ONLY with JSON: {"tool": "...", "args": {...}, "reason": "..."}'
        )
        try:
            data = self.llm.generate_json(self.system_prompt, prompt, step_name="agent_2_plan")
        except Exception as e:
            logger.warning(f"planner LLM call failed, using policy: {type(e).__name__}: {e}")
            self._degrade(f"planner model unavailable ({type(e).__name__})")
            return None
        if not data or data.get("tool") not in TOOLS:
            return None
        args = data.get("args") or {}
        tool = data["tool"]
        if tool == "corpus_search" and not args.get("query"):
            return None
        if tool == "lookup_clause_by_id" and not args.get("clause_id"):
            return None
        if tool == "get_claim_field" and not args.get("field_name"):
            return None
        if tool == "corpus_search" and args.get("query") in state["queries_done"]:
            return None
        return {"tool": tool, "args": args, "reason": str(data.get("reason", ""))[:300], "by": "llm"}

    def _run_tool(self, state: AgentState) -> AgentState:
        action = state["next_action"]
        tool, args = action["tool"], action.get("args", {})
        passages = list(state["passages"])
        pending = list(state["pending_refs"])
        followed = list(state["followed_refs"])
        queries = list(state["queries_done"])
        fields = list(state["fields_checked"])
        update: AgentState = {}
        new: List[Dict[str, Any]] = []
        observation: Any = None

        with span(f"tool.{tool}", {"correlation_id": self.correlation_id, **{k: str(v) for k, v in args.items()}}):
            if tool == "corpus_search":
                strategy = args.get("strategy", "dense")
                if strategy not in ("dense", "keyword", "hybrid"):
                    strategy = "dense"
                new = self.toolbox.corpus_search(query=args["query"], top_k=int(args.get("top_k", 4)), strategy=strategy)
                queries.append(args["query"])
                if len(queries) > 1:  # any search after the first is a follow-up search for modifiers
                    update["modifier_search_done"] = True
                observation = [p["clause_id"] for p in new]
            elif tool == "lookup_clause_by_id":
                cid = self.toolbox.vstore.normalise_clause_id(str(args["clause_id"]))
                hit = self.toolbox.lookup_clause_by_id(cid)
                new = [hit] if hit else []
                followed.append(cid)
                pending = [r for r in pending if r != cid]
                observation = hit["clause_id"] if hit else "not found in corpus"
            elif tool == "get_claim_field":
                field = str(args["field_name"])
                observation = self.toolbox.get_claim_field(field)
                fields.append(field)

        # merge evidence (a clause looked up whole replaces a shorter copy of itself)
        by_clause = {p["clause_id"]: i for i, p in enumerate(passages)}
        for p in new:
            if p["clause_id"] in by_clause:
                i = by_clause[p["clause_id"]]
                if len(p["text"]) > len(passages[i]["text"]):
                    passages[i] = p
            else:
                by_clause[p["clause_id"]] = len(passages)
                passages.append(p)

        # follow references found in newly read text (only in the full agent variant)
        if not self.baseline:
            for p in new:
                for ref in self._extract_refs(p):
                    already = any(rules.clause_compatible(ref, q["clause_id"]) for q in passages)
                    if not already and ref not in pending and ref not in followed and len(pending) < MAX_PENDING_REFS:
                        pending.append(ref)

        update.update(passages=passages, pending_refs=pending, followed_refs=followed,
                      queries_done=queries, fields_checked=fields,
                      _event={"tool": tool, "args": args, "observation": observation,
                              "new_pending_refs": [r for r in pending if r not in state["pending_refs"]]})
        return update

    @staticmethod
    def _extract_refs(passage: Dict[str, Any]) -> List[str]:
        own = passage["clause_id"].upper()
        refs = []
        for m in _REF_RE.finditer(passage["text"]):
            for num in _NUM_RE.findall(m.group(1)):
                parts = num.split(".")
                cid = "CAR-" + ".".join(parts[:3])
                if own.startswith("CAR") and rules.clause_compatible(cid, own):
                    continue
                if cid not in refs:
                    refs.append(cid)
        return refs

    def _decide(self, state: AgentState) -> AgentState:
        h = state["handoff"]
        evidence = rules.Evidence(state["passages"])
        decision = rules.decide(h, evidence, apply_modifiers=not self.baseline)

        if self.demo_fault == "unsupported_citation":
            # Demo item 6: inject a citation the agent never retrieved, to show verification catching it.
            decision["citations"].append({"clause_id": "CAR-3.9.1", "anchor": "additional goodwill compensation of inr 50,000",
                                          "relevance": "Injected unsupported citation (demonstration fault).",
                                          "document_title": rules.CAR_TITLE})

        resolved = []
        for c in decision["citations"]:
            hit = evidence.find(c["clause_id"], c["anchor"])
            resolved.append({**c, "quote": hit[1] if hit else None,
                             "resolved_from": hit[0]["clause_id"] if hit else None})
        decision["citations"] = resolved

        explanation = None
        if decision["routing_decision"] != RoutingDecision.ESCALATE and not self.baseline:
            if self.llm.is_stub:
                try:
                    self.llm.generate(self.system_prompt, "explain", step_name="agent_2_explain")
                except Exception as e:
                    self._degrade(f"explanation model unavailable ({type(e).__name__})")
            else:
                explanation = self._llm_explanation(h, decision)
        if explanation:
            decision["llm_explanation"] = explanation
        return {"decision": decision,
                "_event": {"routing": decision["routing_decision"].value,
                           "compensation_inr": decision["monetary_compensation_inr"],
                           "citations": [c["clause_id"] for c in resolved]}}

    def _llm_explanation(self, h: Dict[str, Any], decision: Dict[str, Any]) -> Optional[str]:
        quotes = "\n".join(f"[{c['clause_id']}] {c['quote']}" for c in decision["citations"] if c.get("quote"))
        prompt = (f"Passenger narrative: {h['free_text_claim_narrative']}\n"
                  f"Determination: {decision['routing_decision'].value}, compensation INR "
                  f"{decision['monetary_compensation_inr']:,.0f}, facilities: {decision['facilities_entitled']}\n"
                  f"Retrieved provisions:\n{quotes}\n\n"
                  "In at most 3 sentences, explain this determination to a customer relations officer. "
                  "Use ONLY the provisions above and cite them by their bracketed ids. Do not change the determination.")
        try:
            text = self.llm.generate(self.system_prompt, prompt, step_name="agent_2_explain").strip()
        except Exception as e:
            logger.warning(f"explanation LLM call failed (degraded to template summary): {type(e).__name__}")
            self._degrade(f"explanation model unavailable ({type(e).__name__})")
            return None
        # A3 for free text: every clause number the model mentions must be one it was given.
        allowed = {c["clause_id"].split("-", 1)[-1] for c in decision["citations"] if c.get("quote")}
        mentioned = set(_NUM_RE.findall(text))
        if any(not any(m == a or m.startswith(a + ".") or a.startswith(m + ".") for a in allowed) for m in mentioned):
            logger.warning("LLM explanation cited an unretrieved clause; discarded by verification")
            return None
        return text[:1200]

    def _verify(self, state: AgentState) -> AgentState:
        d = state["decision"]
        unresolved = [c["clause_id"] for c in d["citations"] if not c.get("quote")]
        problems = [f"citation {c} does not resolve to retrieved text" for c in unresolved]
        if d["routing_decision"] == RoutingDecision.PAY_AS_COMPUTED and d["monetary_compensation_inr"] > 0 \
                and not any(c.get("quote") for c in d["citations"]):
            problems.append("a payable amount is not supported by any retrieved provision")
        if state.get("stop_reason"):
            problems.append(f"run did not complete: {state['stop_reason']}")

        if not problems:
            VERIFICATION_COUNTER.labels(outcome="passed").inc()
            return {"verification": {"route": "pass", "passed": True, "attempts": state["verify_attempts"] + 1},
                    "_event": {"verification": "passed"}}

        attempts = state["verify_attempts"] + 1
        can_retry = (not self.baseline and attempts <= MAX_VERIFY_RETRIES and not state.get("stop_reason")
                     and unresolved and not self._limit_hit(state))
        if can_retry:
            VERIFICATION_COUNTER.labels(outcome="failed_retried").inc()
            RETRY_COUNTER.labels(agent_name="reasoner", kind="verification").inc()
            pending = list(state["pending_refs"])
            followed = list(state["followed_refs"])
            for cid in unresolved:
                if cid not in pending and cid not in followed:
                    pending.append(cid)
            return {"verification": {"route": "retry", "passed": False, "problems": problems, "attempts": attempts},
                    "verify_attempts": attempts, "pending_refs": pending, "strategy": "hybrid",
                    "modifier_search_done": False,
                    "_event": {"verification": "failed", "problems": problems,
                               "retry_strategy": "direct clause lookup + hybrid (keyword+dense) search"}}

        VERIFICATION_COUNTER.labels(outcome="failed_escalated").inc()
        return {"verification": {"route": "escalate", "passed": False, "problems": problems, "attempts": attempts},
                "verify_attempts": attempts, "_event": {"verification": "failed", "problems": problems,
                                                        "action": "escalate"}}

    def _escalate(self, state: AgentState) -> AgentState:
        problems = state["verification"].get("problems", [])
        reason = ("Self-verification failed (A3): " + "; ".join(problems)
                  + ". An unverified answer is not returned; escalated to a customer relations officer.")
        result = {
            "routing_decision": RoutingDecision.ESCALATE,
            "monetary_compensation_inr": 0.0,
            "facilities_entitled": ["Human Customer Relations Officer Review"],
            "refund_entitled": False,
            "clause_citations": [],
            "reasoning_summary": reason,
            "escalation_reason": reason,
            "self_verification_passed": False,
        }
        return {"result": self._with_reconstruction(result, state), "_event": {"escalated": True}}

    def _finalize(self, state: AgentState) -> AgentState:
        d = state["decision"]
        citations = [Citation(clause_id=c["clause_id"], document_title=c["document_title"],
                              quote=c["quote"], relevance=c["relevance"]) for c in d["citations"]]
        summary = d["reasoning_summary"]
        if d.get("llm_explanation"):
            summary += "\n\nAgent explanation: " + d["llm_explanation"]
        if self.degraded_reasons or state["handoff"].get("llm_degraded"):
            summary += ("\n\n[Degraded mode] The language model was unavailable for part of this run, so the "
                        "deterministic fallback planner was used. The answer still passed citation verification.")
        result = {
            "routing_decision": d["routing_decision"],
            "monetary_compensation_inr": d["monetary_compensation_inr"],
            "facilities_entitled": d["facilities_entitled"],
            "refund_entitled": d["refund_entitled"],
            "clause_citations": citations,
            "reasoning_summary": summary,
            "escalation_reason": d.get("escalation_reason"),
            "self_verification_passed": True,
        }
        return {"result": self._with_reconstruction(result, state)}

    def _with_reconstruction(self, result: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        result["agent_trace"] = state.get("trace", [])
        result["retrieved_passages"] = [
            {"clause_id": p["clause_id"], "score": p.get("score"), "text": p["text"]} for p in state["passages"]]
        result["verification"] = state.get("verification", {})
        result["graph_steps"] = state["steps"]
        result["degraded_mode"] = bool(self.degraded_reasons or state["handoff"].get("llm_degraded"))
        result["degraded_reasons"] = self.degraded_reasons + (
            ["classifier model unavailable"] if state["handoff"].get("llm_degraded") else [])
        return result
