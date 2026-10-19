"""
Observability, Metrics and Run Reconstruction Store (D5, A6, F8).
Includes Prometheus metric collectors and persistent run reconstruction logger.
"""

import time
import json
import os
import logging
from typing import Dict, Any, Optional, List
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Set up structured JSON logger
logger = logging.getLogger("claim_triage")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('{"time":"%(asctime)s", "level":"%(levelname)s", "message":"%(message)s"}'))
if not logger.handlers:
    logger.addHandler(handler)

# ---------------------------------------------------------------------------
# Prometheus Metric Declarations (D5, A6)
# ---------------------------------------------------------------------------
CLAIM_SUBMISSIONS_TOTAL = Counter(
    "claim_submissions_total",
    "Total incoming passenger claim submissions",
    ["status"],  # accepted, quarantined, rate_limited
)

CLAIM_RUN_LATENCY = Histogram(
    "claim_run_latency_seconds",
    "End-to-end run processing latency from submission to completion (D6 SLO)",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 45.0],
)

AGENT_STEPS_COUNTER = Counter(
    "agent_steps_total",
    "Total agent reasoning steps executed",
    ["agent_name"],
)

TOOL_CALLS_COUNTER = Counter(
    "tool_calls_total",
    "Total tool invocations across runs",
    ["tool_name"],
)

TOKENS_COUNTER = Counter(
    "llm_tokens_total",
    "Total LLM tokens tracked",
    ["token_type"],  # prompt, completion
)

COST_COUNTER = Counter(
    "llm_cost_usd_total",
    "Cumulative estimated LLM cost in USD",
)

CACHE_COUNTER = Counter(
    "cache_requests_total",
    "Cache lookup results",
    ["result"],  # hit, miss
)

CIRCUIT_TRIPS_COUNTER = Counter(
    "circuit_breaker_trips_total",
    "Number of times the circuit breaker tripped open",
)

UPTIME_GAUGE = Gauge(
    "container_uptime_seconds",
    "Total container uptime in seconds",
)

SERVICE_START_TIME = time.time()


# ---------------------------------------------------------------------------
# Run Reconstruction Store (D5)
# ---------------------------------------------------------------------------
class RunReconstructionStore:
    def __init__(self, runs_dir: str = "data/runs"):
        self.runs_dir = runs_dir
        os.makedirs(self.runs_dir, exist_ok=True)
        self._in_memory_index: Dict[str, Dict[str, Any]] = {}

    def save_run(self, run_id: str, run_data: Dict[str, Any]):
        self._in_memory_index[run_id] = run_data
        file_path = os.path.join(self.runs_dir, f"{run_id}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(run_data, f, indent=2, default=str)

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        if run_id in self._in_memory_index:
            return self._in_memory_index[run_id]
        file_path = os.path.join(self.runs_dir, f"{run_id}.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._in_memory_index[run_id] = data
                return data
        return None

    def list_recent_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        results = list(self._in_memory_index.values())
        return results[-limit:]


def get_prometheus_metrics() -> bytes:
    UPTIME_GAUGE.set(time.time() - SERVICE_START_TIME)
    return generate_latest()
