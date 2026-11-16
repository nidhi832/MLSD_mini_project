"""
Multi-Agent Orchestrator (A1, A2, A3, A5, L4, L5, D5).

Topology (A2):
    API --(claim_processing_queue)--> Agent 1: classifier-agent service
        --(claim_reasoning_queue, structured hand-off)--> Agent 2: agent-worker service (LangGraph)
        --> run record in Redis --> API / web client poll it.

The two agents are separate services with separate prompts, responsibilities and
scaling. This module holds the code for each half (run_classifier / run_reasoner),
plus execute_claim_triage(), which runs both in one process for tests, the
evaluation suite and the no-Redis local mode.

Failure handling: if either agent fails mid-run (exception, provider outage,
circuit open, loop bound hit), the run is completed as an ESCALATE result with the
error recorded — never as a confident answer (A5). If a worker container dies
mid-run, the run record stays in PROCESSING; see docs/RUNBOOK.md.
"""

import os
import time
import uuid
from typing import Any, Dict, Optional

from prompts.registry import get_prompts_by_version
from src.agents.classifier_agent import ClaimClassifierAgent
from src.agents.llm_client import LLMClient
from src.agents.regulatory_agent import RegulatoryReasoningAgent
from src.core.contracts import EntitlementEvaluationResult, PassengerClaimRequest, RoutingDecision
from src.core.observability import AGENT_RUNS_COUNTER, AGENT_STEPS_PER_RUN, AGENT_TOOL_CALLS_PER_RUN
from src.core.tracing import span
from src.guardrails.output_guardrails import OutputGuardrails
from src.ingestion.vector_store import RegulatoryVectorStore
from src.tools.claim_tools import ClaimToolBox


class MultiAgentTriageOrchestrator:
    def __init__(
        self,
        vector_store: RegulatoryVectorStore,
        llm_client: Optional[LLMClient] = None,
        max_steps: Optional[int] = None,
        max_tool_calls: Optional[int] = None,
        wall_clock_timeout_seconds: Optional[float] = None,
    ):
        self.vector_store = vector_store
        self.llm_client = llm_client or LLMClient()
        self.max_steps = max_steps or int(os.getenv("MAX_AGENT_STEPS", "16"))
        self.max_tool_calls = max_tool_calls or int(os.getenv("MAX_AGENT_TOOL_CALLS", "10"))
        self.wall_clock_timeout = wall_clock_timeout_seconds or float(os.getenv("AGENT_WALL_CLOCK_TIMEOUT_SECONDS", "45"))

    # ------------------------------------------------------------------ Agent 1
    def run_classifier(self, claim: PassengerClaimRequest, prompt_version: str = "v2.0",
                       correlation_id: Optional[str] = None, llm: Optional[LLMClient] = None) -> Dict[str, Any]:
        prompts = get_prompts_by_version(prompt_version)
        llm = llm or self.llm_client.for_run()
        t0 = time.time()
        with span("agent1.classify", {"correlation_id": correlation_id, "prompt_version": prompt_version}):
            handoff = ClaimClassifierAgent(llm_client=llm, system_prompt=prompts["classifier"]).process(claim)
        handoff["_agent1"] = {
            "latency_seconds": round(time.time() - t0, 4),
            "cost": llm.cost_tracker.get_totals(),
        }
        AGENT_RUNS_COUNTER.labels(agent_name="classifier", outcome="success").inc()
        return handoff

    # ------------------------------------------------------------------ Agent 2
    def run_reasoner(self, claim: PassengerClaimRequest, handoff: Dict[str, Any], run_id: str,
                     correlation_id: str, prompt_version: str = "v2.0", start_time: Optional[float] = None,
                     llm: Optional[LLMClient] = None, demo_fault: Optional[str] = None) -> EntitlementEvaluationResult:
        start_time = start_time or time.time()
        prompts = get_prompts_by_version(prompt_version)
        llm = llm or self.llm_client.for_run()
        toolbox = ClaimToolBox(claim_request=claim, vector_store=self.vector_store)
        agent1_info = handoff.get("_agent1", {})
        agent1_cost = agent1_info.get("cost", {})

        try:
            with span("agent2.reason", {"correlation_id": correlation_id, "run_id": run_id,
                                        "prompt_version": prompt_version}):
                agent2 = RegulatoryReasoningAgent(
                    llm_client=llm, system_prompt=prompts["reasoner"], toolbox=toolbox,
                    prompt_version=prompt_version, max_steps=self.max_steps, max_tool_calls=self.max_tool_calls,
                    deadline=start_time + self.wall_clock_timeout, correlation_id=correlation_id,
                    demo_fault=demo_fault,
                )
                d = agent2.evaluate_entitlement(handoff)

            # L4 output guardrail: the system may recommend but never authorise a payment
            is_valid, violation, sanitized = OutputGuardrails.filter_output(d["reasoning_summary"])
            routing = d["routing_decision"]
            outcome = "escalated" if routing == RoutingDecision.ESCALATE else "success"
            AGENT_RUNS_COUNTER.labels(agent_name="reasoner", outcome=outcome).inc()
            AGENT_STEPS_PER_RUN.labels(agent_name="reasoner").observe(d.get("graph_steps", 0))
            AGENT_TOOL_CALLS_PER_RUN.labels(agent_name="reasoner").observe(toolbox.get_total_calls())

            return self._build_result(
                claim, handoff, run_id, correlation_id, prompt_version, start_time, llm, toolbox, agent1_cost,
                routing=routing, comp=d["monetary_compensation_inr"], facilities=d["facilities_entitled"],
                refund=d["refund_entitled"], citations=d["clause_citations"], summary=sanitized,
                escalation_reason=d.get("escalation_reason") or (violation if not is_valid else None),
                verified=d.get("self_verification_passed", False), steps=d.get("graph_steps", 0),
                extras={"agent_trace": d.get("agent_trace", []), "retrieved_passages": d.get("retrieved_passages", []),
                        "verification": d.get("verification", {}), "degraded_mode": d.get("degraded_mode", False),
                        "degraded_reasons": d.get("degraded_reasons", [])},
            )
        except Exception as e:
            # A2/A5: mid-run failure -> escalate with the error recorded, never a confident answer
            AGENT_RUNS_COUNTER.labels(agent_name="reasoner", outcome="failed").inc()
            return self._build_result(
                claim, handoff, run_id, correlation_id, prompt_version, start_time, llm, toolbox, agent1_cost,
                routing=RoutingDecision.ESCALATE, comp=0.0, facilities=["Human Case Review"], refund=False,
                citations=[], summary=(f"Graceful degradation: the reasoning agent failed mid-run "
                                       f"({type(e).__name__}: {e}). Escalated to a human operator."),
                escalation_reason=f"System exception mid-run: {type(e).__name__}: {e}", verified=False, steps=0,
                extras={"error": f"{type(e).__name__}: {e}"},
            )

    def _build_result(self, claim, handoff, run_id, correlation_id, prompt_version, start_time, llm, toolbox,
                      agent1_cost, *, routing, comp, facilities, refund, citations, summary, escalation_reason,
                      verified, steps, extras) -> EntitlementEvaluationResult:
        c2 = llm.cost_tracker.get_totals()
        steps_all = list(agent1_cost.get("steps", [])) + list(c2.get("steps", []))
        total_cost = round(sum(s["cost_usd"] for s in steps_all), 6)
        largest = max(steps_all, key=lambda s: s["cost_usd"]) if steps_all else None
        return EntitlementEvaluationResult(
            run_id=run_id,
            correlation_id=correlation_id,
            claim_id=claim.claim_id,
            pnr=claim.pnr,
            disruption_classification=claim.disruption_type,
            extracted_entities={
                "flight_number": handoff.get("flight_number"),
                "operating_carrier": handoff.get("operating_carrier"),
                "block_time_hours": handoff.get("block_time_hours"),
                "notice_hours_before_departure": handoff.get("notice_hours_before_departure"),
                "alternate_flight_offered": handoff.get("offered_alternate_flight"),
                "alternate_flight_accepted": handoff.get("accepted_alternate_flight"),
                "contact_details_at_booking": handoff.get("contact_details_provided_at_booking"),
                "cause_attribution": handoff.get("cause_attribution"),
                "narrative_extraction": handoff.get("narrative_extraction"),
                "fact_conflicts": handoff.get("fact_conflicts", []),
            },
            routing_decision=routing,
            monetary_compensation_inr=comp,
            facilities_entitled=facilities,
            refund_entitled=refund,
            clause_citations=citations,
            reasoning_summary=summary,
            escalation_reason=escalation_reason,
            prompt_version=prompt_version,
            self_verification_passed=verified,
            steps_executed=steps,
            tool_calls_made=toolbox.get_total_calls(),
            tokens_prompt=sum(s["prompt_tokens"] for s in steps_all),
            tokens_completion=sum(s["completion_tokens"] for s in steps_all),
            cost_usd=total_cost,
            execution_time_seconds=round(time.time() - start_time, 4),
            llm_provider=llm.provider,
            llm_model=llm.model,
            tool_call_log=toolbox.tool_call_history,
            cost_breakdown={
                "per_step": steps_all,
                "total_cost_usd": total_cost,
                "cost_per_100_runs_usd": round(total_cost * 100, 4),
                "largest_single_cost_driver": largest["step_name"] if largest else "none",
            },
            **extras,
        )

    # ------------------------------------------------------------------ both, in-process
    def execute_claim_triage(
        self,
        claim: PassengerClaimRequest,
        correlation_id: Optional[str] = None,
        prompt_version: str = "v2.0",
        run_id: Optional[str] = None,
        demo_fault: Optional[str] = None,
    ) -> EntitlementEvaluationResult:
        start = time.time()
        run_id = run_id or f"run-{uuid.uuid4().hex[:12]}"
        corr_id = correlation_id or f"corr-{uuid.uuid4().hex[:8]}"
        try:
            handoff = self.run_classifier(claim, prompt_version=prompt_version, correlation_id=corr_id)
        except Exception as e:
            AGENT_RUNS_COUNTER.labels(agent_name="classifier", outcome="failed").inc()
            return self.escalated_after_agent1_failure(claim, run_id, corr_id, prompt_version, start, e)
        return self.run_reasoner(claim, handoff, run_id, corr_id, prompt_version=prompt_version,
                                 start_time=start, demo_fault=demo_fault)

    def escalated_after_agent1_failure(self, claim, run_id, corr_id, prompt_version, start, error) -> EntitlementEvaluationResult:
        llm = self.llm_client.for_run()
        toolbox = ClaimToolBox(claim_request=claim, vector_store=self.vector_store)
        return self._build_result(
            claim, {}, run_id, corr_id, prompt_version, start, llm, toolbox, {},
            routing=RoutingDecision.ESCALATE, comp=0.0, facilities=["Human Case Review"], refund=False, citations=[],
            summary=f"Agent 1 (classification) failed mid-run ({type(error).__name__}: {error}); escalated to a human.",
            escalation_reason=f"Agent 1 failure: {type(error).__name__}: {error}", verified=False, steps=0,
            extras={"error": f"{type(error).__name__}: {error}"},
        )
