"""
Tests for the LangGraph regulatory agent (A1, A3, A5, F7, L1).
Run: python -m pytest tests/ -v
"""

import json
import re

import pytest

from src.agents.llm_client import LLMClient
from src.agents.orchestrator import MultiAgentTriageOrchestrator
from src.core.cache import ClaimCache
from src.core.contracts import PassengerClaimRequest, RoutingDecision
from src.core.resilience import CircuitBreaker
from src.ingestion.vector_store import RegulatoryVectorStore

SCENARIOS = json.load(open("data/evaluation_set_20.json", encoding="utf-8"))


def _scenario(sid):
    return next(s for s in SCENARIOS if s["id"] == sid)


def _claim(sid, **overrides):
    payload = dict(_scenario(sid)["claim_payload"])
    payload.update(overrides)
    return PassengerClaimRequest.model_validate(payload)


def _norm(t):
    return re.sub(r"\s+", " ", t.lower())


@pytest.fixture(scope="module")
def vstore():
    return RegulatoryVectorStore()


@pytest.fixture
def orch(vstore):
    return MultiAgentTriageOrchestrator(vector_store=vstore, llm_client=LLMClient(provider="stub"))


def test_lookup_returns_the_requested_clause(vstore):
    hit = vstore.lookup_clause("CAR-3.3.2")
    assert hit["clause_id"] == "CAR-3.3.2"
    assert "accepts an alternate flight" in hit["text"]
    merged = vstore.lookup_clause("Clause 3.7")
    assert "Meals and refreshments" in merged["text"]  # 3.7 heading merged with 3.7.1


def test_agent_follows_cross_reference_in_section_4_1(orch):
    """Spec section 5 step 2: the agent must find the provision that modifies the one it retrieved."""
    r = orch.execute_claim_triage(_claim("SCENARIO-01"))
    assert r.routing_decision == RoutingDecision.REQUEST_FURTHER_EVIDENCE
    assert r.monetary_compensation_inr == 0.0
    assert any("meals" in f.lower() for f in r.facilities_entitled)
    looked_up = [c["parameters"].get("clause_id") for c in r.tool_call_log if c["tool"] == "lookup_clause_by_id"]
    assert "CAR-3.7" in looked_up, "agent should follow 3.3.2's reference to the facilities clause 3.7"
    searches = [c for c in r.tool_call_log if c["tool"] == "corpus_search"]
    assert len(searches) >= 2, "agent should search again with a query derived from what it read"
    assert r.self_verification_passed


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["id"] for s in SCENARIOS])
def test_every_citation_resolves_to_retrieved_text(orch, scenario):
    """A3: every citation quote is a line of a passage the agent actually retrieved."""
    r = orch.execute_claim_triage(PassengerClaimRequest.model_validate(scenario["claim_payload"]))
    evidence = _norm(" ".join(p["text"] for p in r.retrieved_passages))
    for c in r.clause_citations:
        assert _norm(c.quote) in evidence, f"{c.clause_id} quote not in retrieved text"
    assert r.routing_decision.value == scenario["expected_routing"]
    assert abs(r.monetary_compensation_inr - scenario["expected_compensation_inr"]) <= 100


def test_verification_catches_unsupported_citation_then_escalates(orch):
    """Demo item 6: an unsupported citation triggers re-search, then escalation; never returned."""
    r = orch.execute_claim_triage(_claim("SCENARIO-04"), demo_fault="unsupported_citation")
    assert r.routing_decision == RoutingDecision.ESCALATE
    assert not r.self_verification_passed
    assert r.monetary_compensation_inr == 0.0
    verify_events = [e for e in r.agent_trace if e["node"] == "verify"]
    assert len(verify_events) >= 2 and verify_events[0]["verification"] == "failed"
    assert "retry_strategy" in verify_events[0], "first failure must trigger a re-search with a different strategy"


def test_baseline_single_pass_misses_the_modifier(orch):
    """The v1.0 baseline (one retrieval, no reference following) gets the Section 4.1 case wrong."""
    r = orch.execute_claim_triage(_claim("SCENARIO-01"), prompt_version="v1.0")
    assert r.tool_call_log and len(r.tool_call_log) == 1
    assert r.routing_decision != RoutingDecision.REQUEST_FURTHER_EVIDENCE


def test_narrative_contradicting_fields_is_escalated(orch):
    """F7: the narrative says the alternate flight was taken, the form says it was not."""
    r = orch.execute_claim_triage(_claim("SCENARIO-01", accepted_alternate_flight=False))
    assert r.routing_decision == RoutingDecision.ESCALATE
    assert r.extracted_entities["fact_conflicts"]


def test_loop_bound_escalates_instead_of_answering(vstore):
    """A1/A5: a run stopped by its tool-call bound did not complete and must not answer."""
    o = MultiAgentTriageOrchestrator(vector_store=vstore, llm_client=LLMClient(provider="stub"), max_tool_calls=1)
    r = o.execute_claim_triage(_claim("SCENARIO-01"))
    assert r.routing_decision == RoutingDecision.ESCALATE
    assert "did not complete" in r.escalation_reason


def test_open_circuit_degrades_to_policy_planner(vstore):
    """A5: with the provider circuit OPEN, the agent falls back to its deterministic planner."""
    cb = CircuitBreaker(name="test_provider_open", recovery_timeout_seconds=999)
    cb.trip_manually()
    llm = LLMClient(provider="groq", model="test-model", circuit_breaker=cb)
    o = MultiAgentTriageOrchestrator(vector_store=vstore, llm_client=llm)
    r = o.execute_claim_triage(_claim("SCENARIO-01"))
    plans = [e["action"] for e in r.agent_trace if e["node"] == "plan"]
    assert all(a["by"] in ("policy", "bound") for a in plans)
    assert r.routing_decision == RoutingDecision.REQUEST_FURTHER_EVIDENCE
    assert r.self_verification_passed


def test_cost_is_tracked_per_run_not_accumulated(orch):
    a = orch.execute_claim_triage(_claim("SCENARIO-02"))
    b = orch.execute_claim_triage(_claim("SCENARIO-02"))
    assert a.cost_usd == b.cost_usd
    assert a.cost_breakdown["per_step"]


def test_cache_key_covers_all_outcome_fields():
    cache = ClaimCache()
    base = _scenario("SCENARIO-04")["claim_payload"]
    assert cache.compute_claim_hash(base) != cache.compute_claim_hash({**base, "contact_details_provided_at_booking": False})
    assert cache.compute_claim_hash(base) == cache.compute_claim_hash({**base, "passenger_name": "Someone Else"})
