"""
Agent worker services (A2, A4, A7).

One image, two roles, two Compose services:

  python -m src.api.worker --role classifier   -> classifier-agent  (Agent 1)
      consumes claim_processing_queue, extracts entities / checks the narrative,
      pushes the structured hand-off to claim_reasoning_queue.

  python -m src.api.worker --role reasoner     -> agent-worker      (Agent 2, LangGraph)
      consumes claim_reasoning_queue, runs the regulatory agent with
      self-verification, writes the full run reconstruction to Redis.

Both are stateless, so `docker compose up --scale agent-worker=3` adds reasoning
capacity. Each exposes Prometheus metrics on WORKER_METRICS_PORT (default 9100).
"""

import argparse
import os
import random
import signal
import time
from typing import Optional

from src.agents.llm_client import LLMClient
from src.agents.orchestrator import MultiAgentTriageOrchestrator
from src.core.cache import ClaimCache
from src.core.contracts import PassengerClaimRequest
from src.core.observability import (
    AGENT_RUNS_COUNTER,
    AGENT_STEP_LATENCY,
    AGENT_STEPS_COUNTER,
    CACHE_COUNTER,
    CLAIM_RUN_LATENCY,
    COST_COUNTER,
    LIVE_QUALITY_GAUGE,
    RETRY_COUNTER,
    RunReconstructionStore,
    TOKENS_COUNTER,
    TOOL_CALLS_COUNTER,
    logger,
    start_worker_metrics_server,
)
from src.core.queue import TaskQueue
from src.core.tracing import extract_context, init_tracing, inject_context, span
from src.eval.metrics import LLMOpsMetrics
from src.ingestion.vector_store import RegulatoryVectorStore

LIVE_EVAL_SAMPLE_RATE = float(os.getenv("LIVE_EVAL_SAMPLE_RATE", "0.25"))

CLASSIFY_QUEUE = os.getenv("CLAIM_QUEUE_NAME", "claim_processing_queue")
REASON_QUEUE = os.getenv("REASONING_QUEUE_NAME", "claim_reasoning_queue")


class ClaimAgentWorker:
    def __init__(
        self,
        role: str = "reasoner",
        worker_id: Optional[str] = None,
        vector_store: Optional[RegulatoryVectorStore] = None,
        cache: Optional[ClaimCache] = None,
        run_store: Optional[RunReconstructionStore] = None,
    ):
        assert role in ("classifier", "reasoner")
        self.role = role
        self.worker_id = worker_id or f"{role}-{os.getenv('HOSTNAME', os.getpid())}"
        self.in_queue = TaskQueue(queue_name=CLASSIFY_QUEUE if role == "classifier" else REASON_QUEUE)
        self.out_queue = TaskQueue(queue_name=REASON_QUEUE) if role == "classifier" else None
        self.cache = cache or ClaimCache()
        self.run_store = run_store or RunReconstructionStore()
        llm = LLMClient()
        agent = "classifier" if role == "classifier" else "reasoner"
        llm.on_retry = lambda attempt: RETRY_COUNTER.labels(agent_name=agent, kind="llm_call").inc()
        # Agent 1 never searches the corpus, so it does not load the embedding model.
        self.orchestrator = MultiAgentTriageOrchestrator(
            vector_store=vector_store or (RegulatoryVectorStore() if role == "reasoner" else None),
            llm_client=llm,
        )
        self.is_running = True

    def start(self):
        logger.info(f"Worker {self.worker_id} ({self.role}) listening on '{self.in_queue.queue_name}' "
                    f"(redis={self.in_queue.is_using_redis()})")

        def _stop(signum, frame):
            logger.info(f"Worker {self.worker_id} received signal {signum}; finishing current task then stopping")
            self.is_running = False

        signal.signal(signal.SIGINT, _stop)
        signal.signal(signal.SIGTERM, _stop)

        while self.is_running:
            try:
                task = self.in_queue.dequeue(timeout_seconds=2)
                if task:
                    self.process_task(task)
            except Exception as e:
                logger.error(f"Worker {self.worker_id} loop error: {e}")
                time.sleep(0.5)
        logger.info(f"Worker {self.worker_id} stopped")

    def process_task(self, task: dict):
        if self.role == "classifier":
            self._classify(task)
        else:
            self._reason(task)

    # ------------------------------------------------------------------ Agent 1
    def _classify(self, task: dict):
        run_id, corr_id = task["run_id"], task["correlation_id"]
        prompt_version = task.get("prompt_version", "v2.0")
        ctx = extract_context(task.get("trace_context"))
        with span("worker.classifier", {"run_id": run_id, "correlation_id": corr_id}, context=ctx):
            t0 = time.time()
            self.run_store.save_run(run_id, {"run_id": run_id, "correlation_id": corr_id, "status": "CLASSIFYING",
                                             "classifier_worker": self.worker_id})
            try:
                claim = PassengerClaimRequest.model_validate(task["claim_data"])
                claim_hash = self.cache.compute_claim_hash(task["claim_data"], prompt_version)
                if not task.get("demo_fault"):
                    cached = self.cache.get_run(claim_hash)
                    if cached:
                        CACHE_COUNTER.labels(result="hit").inc()
                        cached.update(run_id=run_id, correlation_id=corr_id, is_cached=True)
                        duration = time.time() - task.get("submitted_at", t0)
                        CLAIM_RUN_LATENCY.observe(duration)
                        self.run_store.save_run(run_id, {"status": "COMPLETED", "result": cached, "cached": True,
                                                         "completed_at": time.time(),
                                                         "duration_seconds": round(duration, 4)})
                        logger.info(f"cache hit run_id={run_id} corr_id={corr_id}")
                        return
                    CACHE_COUNTER.labels(result="miss").inc()

                handoff = self.orchestrator.run_classifier(claim, prompt_version=prompt_version, correlation_id=corr_id)
                AGENT_STEPS_COUNTER.labels(agent_name="classifier").inc()
                AGENT_STEP_LATENCY.labels(agent_name="classifier", step="classify").observe(time.time() - t0)
                handoff["disruption_type"] = claim.disruption_type.value
                self.run_store.save_run(run_id, {"status": "QUEUED_FOR_REASONING",
                                                 "agent1_handoff": {k: v for k, v in handoff.items() if k != "_agent1"}})
                self.out_queue.enqueue({**task, "handoff": handoff, "claim_hash": claim_hash,
                                        "trace_context": inject_context()})
                logger.info(f"handoff run_id={run_id} corr_id={corr_id} conflicts={len(handoff.get('fact_conflicts', []))}")
            except Exception as e:
                AGENT_RUNS_COUNTER.labels(agent_name="classifier", outcome="failed").inc()
                logger.error(f"Agent 1 failure run_id={run_id} corr_id={corr_id}: {e}")
                claim = PassengerClaimRequest.model_validate(task["claim_data"])
                result = self.orchestrator.escalated_after_agent1_failure(
                    claim, run_id, corr_id, prompt_version, task.get("submitted_at", t0), e)
                self.run_store.save_run(run_id, {"status": "COMPLETED", "result": result.model_dump(mode="json"),
                                                 "completed_at": time.time()})

    # ------------------------------------------------------------------ Agent 2
    def _reason(self, task: dict):
        run_id, corr_id = task["run_id"], task["correlation_id"]
        prompt_version = task.get("prompt_version", "v2.0")
        ctx = extract_context(task.get("trace_context"))
        with span("worker.reasoner", {"run_id": run_id, "correlation_id": corr_id}, context=ctx):
            submitted = task.get("submitted_at", time.time())
            self.run_store.save_run(run_id, {"status": "PROCESSING", "worker_id": self.worker_id,
                                             "started_at": time.time()})
            try:
                claim = PassengerClaimRequest.model_validate(task["claim_data"])
                result = self.orchestrator.run_reasoner(
                    claim, task["handoff"], run_id, corr_id, prompt_version=prompt_version,
                    start_time=time.time(), demo_fault=task.get("demo_fault"))
                result_dict = result.model_dump(mode="json")
                duration = time.time() - submitted  # D6: submission to final result, not the submit call
                result_dict["end_to_end_seconds"] = round(duration, 4)

                if not task.get("demo_fault") and result.self_verification_passed:
                    self.cache.set_run(task.get("claim_hash") or self.cache.compute_claim_hash(task["claim_data"], prompt_version),
                                       result_dict)

                CLAIM_RUN_LATENCY.observe(duration)
                AGENT_STEPS_COUNTER.labels(agent_name="reasoner").inc(result.steps_executed)
                for call in result.tool_call_log:
                    TOOL_CALLS_COUNTER.labels(tool_name=call["tool"]).inc()
                TOKENS_COUNTER.labels(token_type="prompt").inc(result.tokens_prompt)
                TOKENS_COUNTER.labels(token_type="completion").inc(result.tokens_completion)
                COST_COUNTER.inc(result.cost_usd)

                live_scores = None
                if random.random() < LIVE_EVAL_SAMPLE_RATE:
                    live_scores = self._score_live_run(task["claim_data"], result_dict)

                self.run_store.save_run(run_id, {"status": "COMPLETED", "result": result_dict, "cached": False,
                                                 "completed_at": time.time(), "duration_seconds": round(duration, 4),
                                                 "live_evaluation": live_scores})
                logger.info(f"completed run_id={run_id} corr_id={corr_id} decision={result.routing_decision.value} "
                            f"verified={result.self_verification_passed} duration={duration:.2f}s")
            except Exception as e:
                AGENT_RUNS_COUNTER.labels(agent_name="reasoner", outcome="failed").inc()
                logger.error(f"run failure run_id={run_id} corr_id={corr_id}: {e}")
                self.run_store.save_run(run_id, {"status": "FAILED", "error": str(e), "completed_at": time.time()})


    @staticmethod
    def _score_live_run(claim_data: dict, result: dict) -> dict:
        """L6: automatically score a sample of live runs with the evaluation metrics (no labels needed)."""
        passages = result.get("retrieved_passages", [])
        scores = {
            "groundedness": LLMOpsMetrics.compute_groundedness(result, passages),
            "context_relevance": LLMOpsMetrics.compute_context_relevance(claim_data, passages),
            "verification_pass": 1.0 if result.get("self_verification_passed") else 0.0,
        }
        for metric, value in scores.items():
            LIVE_QUALITY_GAUGE.labels(metric=metric).set(value)
        return scores


def main():
    parser = argparse.ArgumentParser(description="Claim triage agent worker")
    parser.add_argument("--role", choices=["classifier", "reasoner"], default=os.getenv("AGENT_ROLE", "reasoner"))
    args = parser.parse_args()
    os.environ.setdefault("OTEL_SERVICE_NAME", "classifier-agent" if args.role == "classifier" else "agent-worker")
    init_tracing()
    start_worker_metrics_server(int(os.getenv("WORKER_METRICS_PORT", "9100")))
    ClaimAgentWorker(role=args.role).start()


if __name__ == "__main__":
    main()
