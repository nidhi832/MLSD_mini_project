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
handler.setFormatter(logging.Formatter(
    '{"time":"%(asctime)s", "level":"%(levelname)s", "service":"' + os.getenv("OTEL_SERVICE_NAME", "claim-triage-service") + '", "message":"%(message)s"}'
))
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
    ["service"],
)

# Per-agent metrics (A6): success rate, steps, tool calls, per-step latency, retries
AGENT_RUNS_COUNTER = Counter(
    "agent_runs_total",
    "Agent executions by outcome",
    ["agent_name", "outcome"],  # outcome: success, escalated, failed
)

AGENT_STEP_LATENCY = Histogram(
    "agent_step_latency_seconds",
    "Latency of each agent step / graph node",
    ["agent_name", "step"],
    buckets=[0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0],
)

AGENT_STEPS_PER_RUN = Histogram(
    "agent_steps_per_run",
    "Number of graph steps taken per run",
    ["agent_name"],
    buckets=[1, 2, 3, 4, 5, 6, 8, 10, 12, 16],
)

AGENT_TOOL_CALLS_PER_RUN = Histogram(
    "agent_tool_calls_per_run",
    "Number of tool calls made per run",
    ["agent_name"],
    buckets=[0, 1, 2, 3, 4, 5, 6, 8, 10, 12],
)

RETRY_COUNTER = Counter(
    "agent_retries_total",
    "Retries: LLM call retries and verification-driven re-searches",
    ["agent_name", "kind"],  # kind: llm_call, verification
)

VERIFICATION_COUNTER = Counter(
    "agent_verification_total",
    "Self-verification outcomes (A3)",
    ["outcome"],  # passed, failed_retried, failed_escalated
)

LIVE_QUALITY_GAUGE = Gauge(
    "live_run_quality_score",
    "Automatic quality score of the most recent sampled live runs (L6)",
    ["metric"],  # groundedness, citation_resolution
)

FEEDBACK_SATISFACTION_GAUGE = Gauge("feedback_satisfaction_rate", "Share of analyst feedback marked satisfactory (L6)")

# Data freshness and ingestion pipeline metrics (D2)
INDEX_CHUNKS_GAUGE = Gauge("vector_index_total_chunks", "Chunks in the vector index")
INDEX_BUILD_TIMESTAMP_GAUGE = Gauge("vector_index_build_timestamp_seconds", "Unix time the index was built")
INDEX_AGE_GAUGE = Gauge("vector_index_age_seconds", "Seconds since the index was built (data freshness)")
INGESTION_THROUGHPUT_GAUGE = Gauge("ingestion_throughput_chunks_per_second", "Throughput of the last ingestion run")


def publish_index_metrics(manifest: Dict[str, Any]) -> None:
    import calendar
    try:
        INDEX_CHUNKS_GAUGE.set(manifest.get("total_chunks", 0))
        INGESTION_THROUGHPUT_GAUGE.set(manifest.get("throughput_chunks_per_second", 0))
        ts = manifest.get("build_timestamp")
        if ts:
            built = calendar.timegm(time.strptime(ts, "%Y-%m-%dT%H:%M:%SZ"))
            INDEX_BUILD_TIMESTAMP_GAUGE.set(built)
            INDEX_AGE_GAUGE.set_function(lambda: time.time() - built)
    except Exception:
        pass


SERVICE_START_TIME = time.time()
SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "claim-triage-service")


# ---------------------------------------------------------------------------
# Run Reconstruction Store (D5)
# ---------------------------------------------------------------------------
class RunReconstructionStore:
    """
    Stores the full reconstruction of each run, keyed by run_id.

    Backed by Redis so that the API container can read what the agent-worker
    containers wrote (they do not share a filesystem). Every write is also
    mirrored to data/runs/<run_id>.json in the writing container for offline
    inspection; without Redis (tests, local runs) the file store is used alone.
    Writes MERGE into the existing record, so status updates from different
    services do not erase each other's fields.
    """

    KEY_PREFIX = "run:"
    INDEX_KEY = "runs:index"
    TTL_SECONDS = 7 * 24 * 3600

    def __init__(self, runs_dir: str = "data/runs"):
        self.runs_dir = runs_dir
        os.makedirs(self.runs_dir, exist_ok=True)
        self._in_memory_index: Dict[str, Dict[str, Any]] = {}

    def _redis(self):
        from src.core.shared_state import get_redis
        return get_redis()

    def save_run(self, run_id: str, run_data: Dict[str, Any]):
        existing = self.get_run(run_id) or {}
        merged = {**existing, **run_data}
        merged.setdefault("run_id", run_id)
        merged["updated_at"] = time.time()
        payload = json.dumps(merged, default=str)

        client = self._redis()
        if client is not None:
            try:
                client.set(self.KEY_PREFIX + run_id, payload, ex=self.TTL_SECONDS)
                client.zadd(self.INDEX_KEY, {run_id: merged["updated_at"]})
            except Exception as e:
                logger.error(f"run store redis write failed run_id={run_id}: {e}")

        self._in_memory_index[run_id] = json.loads(payload)
        file_path = os.path.join(self.runs_dir, f"{run_id}.json")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(merged, indent=2, default=str))
        except OSError:
            pass

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        client = self._redis()
        if client is not None:
            try:
                raw = client.get(self.KEY_PREFIX + run_id)
                if raw:
                    return json.loads(raw)
            except Exception:
                pass
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
        client = self._redis()
        if client is not None:
            try:
                ids = client.zrevrange(self.INDEX_KEY, 0, limit - 1)
                return [r for r in (self.get_run(i) for i in ids) if r]
            except Exception:
                pass
        results = list(self._in_memory_index.values())
        return results[-limit:]


def get_prometheus_metrics() -> bytes:
    UPTIME_GAUGE.labels(service=SERVICE_NAME).set(time.time() - SERVICE_START_TIME)
    return generate_latest()


def start_worker_metrics_server(port: int) -> None:
    """Expose /metrics from a worker process so Prometheus can scrape every service (D5)."""
    import threading
    from prometheus_client import start_http_server

    start_http_server(port)

    def _tick():
        while True:
            UPTIME_GAUGE.labels(service=SERVICE_NAME).set(time.time() - SERVICE_START_TIME)
            time.sleep(5)

    threading.Thread(target=_tick, daemon=True).start()
