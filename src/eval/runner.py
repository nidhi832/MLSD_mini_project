"""
Evaluation batch runner (L2, L3).

    python -m src.eval.runner                 # evaluate v2.0 on the 20 scenarios
    python -m src.eval.runner --compare       # v1.0 vs v2.0, all four metrics before/after
    python -m src.eval.runner --compare --out data/runs/eval_report.json

The two variants are real configurations of the same agent, run on the same set:
  v1.0 (before): baseline prompt; one retrieval pass, answer from what came back,
                 no reference following, no re-search when verification fails.
  v2.0 (after):  step-back prompt; agentic loop that follows cross-references,
                 searches again with a query derived from what it read, and
                 re-searches / escalates when verification fails.
The deliberate change between them is enabling steps 2-5 of spec section 5.
"""

import argparse
import json
import os
from typing import Any, Dict

from src.agents.llm_client import LLMClient
from src.agents.orchestrator import MultiAgentTriageOrchestrator
from src.core.contracts import PassengerClaimRequest
from src.eval.metrics import LLMOpsMetrics
from src.ingestion.vector_store import RegulatoryVectorStore

METRICS = ["groundedness", "context_relevance", "faithfulness", "hallucination_rate", "supporting_passage_recall"]


def run_evaluation_suite(eval_file: str = "data/evaluation_set_20.json", prompt_version: str = "v2.0",
                         orchestrator: MultiAgentTriageOrchestrator = None) -> Dict[str, Any]:
    with open(eval_file, "r", encoding="utf-8") as f:
        scenarios = json.load(f)
    orch = orchestrator or MultiAgentTriageOrchestrator(vector_store=RegulatoryVectorStore(), llm_client=LLMClient())

    rows = []
    for item in scenarios:
        claim = PassengerClaimRequest.model_validate(item["claim_payload"])
        res = orch.execute_claim_triage(claim, prompt_version=prompt_version).model_dump(mode="json")
        passages = res.get("retrieved_passages", [])
        g = LLMOpsMetrics.compute_groundedness(res, passages)
        rows.append({
            "id": item["id"],
            "title": item["title"],
            "expected_routing": item["expected_routing"],
            "actual_routing": res["routing_decision"],
            "expected_comp": item["expected_compensation_inr"],
            "actual_comp": res["monetary_compensation_inr"],
            "groundedness": g,
            "context_relevance": LLMOpsMetrics.compute_context_relevance(item["claim_payload"], passages),
            "faithfulness": LLMOpsMetrics.compute_faithfulness(res, item),
            "hallucination": LLMOpsMetrics.is_hallucination(res, item, g),
            "supporting_passage_recall": LLMOpsMetrics.supporting_passage_retrieved(item, passages),
            "tool_calls": res["tool_calls_made"],
            "cost_usd": res["cost_usd"],
            "latency_s": res["execution_time_seconds"],
        })

    n = len(rows)
    avg = lambda k: round(sum(float(r[k]) for r in rows) / n, 4)  # noqa: E731
    return {
        "prompt_version": prompt_version,
        "llm_provider": orch.llm_client.provider,
        "scenarios_evaluated": n,
        "groundedness": avg("groundedness"),
        "context_relevance": avg("context_relevance"),
        "faithfulness": avg("faithfulness"),
        "hallucination_rate": round(sum(r["hallucination"] for r in rows) / n, 4),
        "supporting_passage_recall": avg("supporting_passage_recall"),
        "correct_outcomes": sum(1 for r in rows if r["faithfulness"] == 1.0),
        "avg_tool_calls": avg("tool_calls"),
        "cost_per_100_runs_usd": round(sum(r["cost_usd"] for r in rows) / n * 100, 4),
        "scenario_results": rows,
    }


def _print_single(r: Dict[str, Any]):
    print(f"Variant {r['prompt_version']} (LLM provider: {r['llm_provider']}), {r['scenarios_evaluated']} scenarios")
    for m in METRICS:
        print(f"  {m:<26}: {r[m]}")
    print(f"  correct outcomes          : {r['correct_outcomes']}/{r['scenarios_evaluated']}")
    print(f"  avg tool calls per run    : {r['avg_tool_calls']}")
    print(f"  cost per 100 runs (USD)   : {r['cost_per_100_runs_usd']}")
    wrong = [s for s in r["scenario_results"] if s["faithfulness"] < 1.0]
    for s in wrong:
        print(f"    x {s['id']}: expected {s['expected_routing']} INR {s['expected_comp']:,.0f}, "
              f"got {s['actual_routing']} INR {s['actual_comp']:,.0f}")


def main():
    parser = argparse.ArgumentParser(description="Run the 20-scenario evaluation suite (L2, L3)")
    parser.add_argument("--eval-file", default="data/evaluation_set_20.json")
    parser.add_argument("--compare", action="store_true", help="Compare v1.0 (before) with v2.0 (after)")
    parser.add_argument("--out", default=None, help="Write the full report as JSON to this path")
    args = parser.parse_args()

    orch = MultiAgentTriageOrchestrator(vector_store=RegulatoryVectorStore(), llm_client=LLMClient())
    print("=" * 74)
    print("AIR PASSENGER CLAIM TRIAGE - EVALUATION SUITE (L2, L3)")
    print("=" * 74)

    report: Dict[str, Any]
    if args.compare:
        before = run_evaluation_suite(args.eval_file, "v1.0", orch)
        after = run_evaluation_suite(args.eval_file, "v2.0", orch)
        print(f"{'Metric':<27}| {'Before (v1.0)':<14}| {'After (v2.0)':<14}| Change")
        print("-" * 74)
        for m in METRICS:
            d = round(after[m] - before[m], 4)
            print(f"{m:<27}| {before[m]:<14}| {after[m]:<14}| {'+' if d > 0 else ''}{d}")
        print(f"{'correct outcomes':<27}| {before['correct_outcomes']:<14}| {after['correct_outcomes']:<14}|")
        print(f"{'avg tool calls per run':<27}| {before['avg_tool_calls']:<14}| {after['avg_tool_calls']:<14}|")
        print("-" * 74)
        print("Deliberate change: enable reference following, a derived second search and")
        print("verification-driven re-search (spec section 5, steps 2-5).")
        for label, r in (("before", before), ("after", after)):
            wrong = [s["id"] for s in r["scenario_results"] if s["faithfulness"] < 1.0]
            print(f"  wrong outcomes {label}: {', '.join(wrong) or 'none'}")
        report = {"before": before, "after": after}
    else:
        report = run_evaluation_suite(args.eval_file, "v2.0", orch)
        _print_single(report)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Full report written to {args.out}")


if __name__ == "__main__":
    main()
