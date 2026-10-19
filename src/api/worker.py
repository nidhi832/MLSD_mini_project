"""
Agent Worker Service (A4, A7).
Stateless worker that continuously consumes claims from the task queue,
invokes the multi-agent orchestrator, persists the run reconstruction, and updates cache and metrics.
Can be scaled arbitrarily via: docker compose up --scale agent-worker=3
"""

import os
import sys
import time
import signal
import json
from typing import Optional

from src.core.contracts import PassengerClaimRequest, RoutingDecision
from src.core.queue import TaskQueue
from src.core.cache import ClaimCache
from src.core.observability import (
    RunReconstructionStore,
    CLAIM_RUN_LATENCY,
    AGENT_STEPS_COUNTER,
    TOOL_CALLS_COUNTER,
    TOKENS_COUNTER,
    COST_COUNTER,
    CACHE_COUNTER,
    logger,
)
from src.ingestion.vector_store import RegulatoryVectorStore
from src.agents.orchestrator import MultiAgentTriageOrchestrator
from src.agents.llm_client import LLMClient


class ClaimAgentWorker:
    def __init__(
        self,
        worker_id: Optional[str] = None,
        queue: Optional[TaskQueue] = None,
        vector_store: Optional[RegulatoryVectorStore] = None,
        cache: Optional[ClaimCache] = None,
        run_store: Optional[RunReconstructionStore] = None,
    ):
        self.worker_id = worker_id or f"worker-{os.getpid()}"
        self.queue = queue or TaskQueue()
        self.vector_store = vector_store or RegulatoryVectorStore()
        self.cache = cache or ClaimCache()
        self.run_store = run_store or RunReconstructionStore()

        self.orchestrator = MultiAgentTriageOrchestrator(
            vector_store=self.vector_store,
            llm_client=LLMClient(),
        )
        self.is_running = True

    def start(self):
        logger.info(f"[*] Claim Agent Worker '{self.worker_id}' started. Listening on queue '{self.queue.queue_name}'...")

        def _handle_sigterm(signum, frame):
            logger.info(f"[!] Worker '{self.worker_id}' received termination signal. Shutting down gracefully...")
            self.is_running = False

        signal.signal(signal.SIGINT, _handle_sigterm)
        signal.signal(signal.SIGTERM, _handle_sigterm)

        while self.is_running:
            try:
                task = self.queue.dequeue(timeout_seconds=2)
                if not task:
                    continue

                self.process_task(task)
            except Exception as e:
                logger.error(f"[!] Error processing task in worker '{self.worker_id}': {e}")
                time.sleep(0.5)

        logger.info(f"[+] Worker '{self.worker_id}' gracefully stopped.")

    def process_task(self, task: dict):
        run_id = task["run_id"]
        corr_id = task["correlation_id"]
        claim_raw = task["claim_data"]
        prompt_version = task.get("prompt_version", "v2.0")

        logger.info(f"[*] Worker '{self.worker_id}' processing run_id={run_id} corr_id={corr_id}")

        # Update run status to PROCESSING
        self.run_store.save_run(run_id, {
            "run_id": run_id,
            "correlation_id": corr_id,
            "status": "PROCESSING",
            "worker_id": self.worker_id,
            "started_at": time.time(),
        })

        start_time = time.time()
        try:
            claim_obj = PassengerClaimRequest.model_validate(claim_raw)

            # Check whole-run cache (D7)
            claim_hash = self.cache.compute_claim_hash(claim_raw)
            cached_result = self.cache.get_run(claim_hash)
            if cached_result:
                CACHE_COUNTER.labels(result="hit").inc()
                cached_result["run_id"] = run_id
                cached_result["correlation_id"] = corr_id
                cached_result["is_cached"] = True
                self.run_store.save_run(run_id, {
                    "status": "COMPLETED",
                    "result": cached_result,
                    "worker_id": self.worker_id,
                    "cached": True,
                })
                logger.info(f"[+] Cache hit for run_id={run_id}")
                return

            CACHE_COUNTER.labels(result="miss").inc()

            # Execute reasoning layer
            result = self.orchestrator.execute_claim_triage(
                claim=claim_obj,
                correlation_id=corr_id,
                prompt_version=prompt_version,
            )

            result_dict = result.model_dump(mode="json")
            result_dict["run_id"] = run_id

            # Save in whole-run cache
            self.cache.set_run(claim_hash, result_dict)

            # Record Observability metrics (D5, A6)
            duration = time.time() - start_time
            CLAIM_RUN_LATENCY.observe(duration)
            AGENT_STEPS_COUNTER.labels(agent_name="classifier").inc()
            AGENT_STEPS_COUNTER.labels(agent_name="reasoner").inc()
            TOOL_CALLS_COUNTER.labels(tool_name="corpus_search").inc(result.tool_calls_made)
            TOKENS_COUNTER.labels(token_type="prompt").inc(result.tokens_prompt)
            TOKENS_COUNTER.labels(token_type="completion").inc(result.tokens_completion)
            COST_COUNTER.inc(result.cost_usd)

            # Save full run reconstruction
            self.run_store.save_run(run_id, {
                "status": "COMPLETED",
                "result": result_dict,
                "worker_id": self.worker_id,
                "cached": False,
                "completed_at": time.time(),
                "duration_seconds": round(duration, 4),
            })

            logger.info(f"[+] Completed run_id={run_id} decision={result.routing_decision} duration={duration:.2f}s")

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"[!] Run failure run_id={run_id}: {e}")
            self.run_store.save_run(run_id, {
                "status": "FAILED",
                "error": str(e),
                "worker_id": self.worker_id,
                "completed_at": time.time(),
                "duration_seconds": round(duration, 4),
            })


if __name__ == "__main__":
    worker = ClaimAgentWorker()
    worker.start()
