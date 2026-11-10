"""
Evaluation metrics (L3). Each metric is computed per scenario from the run's own
reconstruction (the passages the agent actually retrieved), then averaged.

1. Groundedness       = cited quotes found verbatim in the retrieved passages / citations.
                         A determination with no citations scores 1.0 only if it is an escalation.
2. Context relevance  = retrieved passages from the regulatory section that governs the
                         disruption (or the facilities / escalation / charter sections) / passages retrieved.
3. Faithfulness       = 0.7 * (routing matches the expected, passage-supported outcome)
                       + 0.3 * (compensation within INR 100 of expected).
4. Hallucination rate = share of scenarios where the system gave a confident (non-escalated)
                         answer that is wrong or cites text it did not retrieve.
                         Escalating is not a hallucination: it is the safe answer.
Also reported: supporting-passage recall = expected supporting clause was retrieved.
"""

import re
from typing import Any, Dict, List

SECTION_FOR = {
    "cancellation": ("CAR-3.3",),
    "delay": ("CAR-3.4",),
    "denied_boarding": ("CAR-3.2",),
    "downgrading": ("CAR-3.5",),
}
ALWAYS_RELEVANT = ("CAR-3.6", "CAR-3.7", "CAR-3.8", "CAR-2", "INDIGO-")


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").lower()).strip()


def _same_clause(a: str, b: str) -> bool:
    a, b = a.upper(), b.upper()
    return a == b or a.startswith(b + ".") or b.startswith(a + ".")


class LLMOpsMetrics:
    @staticmethod
    def compute_groundedness(result: Dict[str, Any], passages: List[Dict[str, Any]]) -> float:
        citations = result.get("clause_citations", [])
        if not citations:
            return 1.0 if str(result.get("routing_decision")) in ("escalate", "RoutingDecision.ESCALATE") else 0.0
        evidence = _norm(" ".join(p.get("text", "") for p in passages))
        grounded = sum(1 for c in citations if c.get("quote") and _norm(c["quote"]) in evidence)
        return round(grounded / len(citations), 4)

    @staticmethod
    def compute_context_relevance(claim: Dict[str, Any], passages: List[Dict[str, Any]]) -> float:
        if not passages:
            return 0.0
        prefixes = SECTION_FOR.get(claim.get("disruption_type", ""), ()) + ALWAYS_RELEVANT
        relevant = sum(1 for p in passages if p.get("clause_id", "").upper().startswith(prefixes))
        return round(relevant / len(passages), 4)

    @staticmethod
    def supporting_passage_retrieved(expected: Dict[str, Any], passages: List[Dict[str, Any]]) -> float:
        target = expected.get("supporting_clause")
        if not target:
            return 1.0
        return 1.0 if any(_same_clause(target, p.get("clause_id", "")) for p in passages) else 0.0

    @staticmethod
    def compute_faithfulness(result: Dict[str, Any], expected: Dict[str, Any]) -> float:
        route_match = 1.0 if _route(result) == expected.get("expected_routing") else 0.0
        diff = abs(result.get("monetary_compensation_inr", 0.0) - expected.get("expected_compensation_inr", 0.0))
        comp_match = 1.0 if diff <= 100.0 else max(0.0, 1.0 - diff / max(expected.get("expected_compensation_inr", 0.0), 5000.0))
        return round(route_match * 0.7 + comp_match * 0.3, 4)

    @staticmethod
    def is_hallucination(result: Dict[str, Any], expected: Dict[str, Any], groundedness: float) -> bool:
        route = _route(result)
        if route == "escalate":
            return False
        wrong = route != expected.get("expected_routing") or abs(
            result.get("monetary_compensation_inr", 0.0) - expected.get("expected_compensation_inr", 0.0)) > 100
        return wrong or groundedness < 1.0


def _route(result: Dict[str, Any]) -> str:
    r = result.get("routing_decision")
    return getattr(r, "value", str(r))
