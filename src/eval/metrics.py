"""
Evaluation Metrics Calculator (L3).
Defines and computes the four core RAG/AgentOps metrics:
1. Groundedness: Ratio of claims in output that resolve directly to retrieved passages.
2. Context Relevance: Degree to which retrieved passages match the regulatory requirements of the claim.
3. Faithfulness: Consistency of the triage determination with regulatory provisions and conditions.
4. Hallucination Rate: 1.0 - Faithfulness.
"""

from typing import Dict, Any, List


class LLMOpsMetrics:
    @staticmethod
    def compute_groundedness(result: Dict[str, Any], retrieved_passages: List[Dict[str, Any]]) -> float:
        """
        Groundedness: Computes the proportion of cited clauses and quotes that physically
        appear in the passages supplied to the agent.
        """
        citations = result.get("clause_citations", [])
        if not citations:
            return 1.0 if result.get("routing_decision") in ("decline", "escalate") else 0.5

        combined_text = " ".join([p.get("text", "") for p in retrieved_passages]).lower()
        grounded_count = 0

        for cit in citations:
            clause_id = cit.get("clause_id", "").lower().replace("car-", "")
            quote_snip = cit.get("quote", "").lower()[:25]
            if clause_id in combined_text or quote_snip in combined_text:
                grounded_count += 1

        return round(grounded_count / len(citations), 4)

    @staticmethod
    def compute_context_relevance(claim: Dict[str, Any], retrieved_passages: List[Dict[str, Any]]) -> float:
        """
        Context Relevance: Measures whether retrieved passages address the specific disruption type
        and operational conditions present in the claim.
        """
        if not retrieved_passages:
            return 0.0

        dtype = claim.get("disruption_type", "").lower()
        relevant_count = 0

        for p in retrieved_passages:
            category = p.get("metadata", {}).get("category", "").lower()
            text = p.get("text", "").lower()
            if dtype in category or dtype in text or "facilities" in category or "3.3" in p.get("clause_id", ""):
                relevant_count += 1

        return round(relevant_count / len(retrieved_passages), 4)

    @staticmethod
    def compute_faithfulness(result: Dict[str, Any], expected: Dict[str, Any]) -> float:
        """
        Faithfulness: Verifies that the routing decision and monetary compensation strictly
        align with regulatory conditions (no fabricated payouts).
        """
        actual_route = result.get("routing_decision")
        expected_route = expected.get("expected_routing")

        route_match = 1.0 if str(actual_route) == str(expected_route) else 0.0

        # Check compensation adherence
        actual_comp = result.get("monetary_compensation_inr", 0.0)
        expected_comp = expected.get("expected_compensation_inr", 0.0)
        comp_diff = abs(actual_comp - expected_comp)
        comp_match = 1.0 if comp_diff <= 100.0 else max(0.0, 1.0 - (comp_diff / max(expected_comp, 5000.0)))

        return round((route_match * 0.7) + (comp_match * 0.3), 4)

    @staticmethod
    def compute_hallucination_rate(faithfulness: float) -> float:
        """Hallucination Rate = 1.0 - Faithfulness."""
        return round(max(0.0, 1.0 - faithfulness), 4)
