"""
FastAPI REST API Routes (F1, F2, F8, A4, D1, D5, D7, L4, L6).
"""

import uuid
import time
from typing import Dict, Any, Optional
from fastapi import APIRouter, Request, HTTPException, Response, Query
from pydantic import ValidationError

from src.core.contracts import PassengerClaimRequest
from src.core.quarantine import QuarantineStore
from src.core.queue import TaskQueue
from src.core.rate_limiter import TokenBucketRateLimiter
from src.core.feedback import FeedbackStore, FeedbackSubmission
from src.core.observability import (
    RunReconstructionStore,
    CLAIM_SUBMISSIONS_TOTAL,
    CLAIM_RUN_LATENCY,
    get_prometheus_metrics,
    logger,
    publish_index_metrics,
    FEEDBACK_SATISFACTION_GAUGE,
)
from src.core.cache import ClaimCache
from src.core.tracing import inject_context, span
from src.guardrails.input_guardrails import InputGuardrails
from src.agents.orchestrator import MultiAgentTriageOrchestrator
from src.ingestion.vector_store import RegulatoryVectorStore
from src.agents.llm_client import LLMClient

router = APIRouter(prefix="/api/v1")

# Singletons for service dependencies
quarantine_store = QuarantineStore()
import os
task_queue = TaskQueue(queue_name=os.getenv("CLAIM_QUEUE_NAME", "claim_processing_queue"))
rate_limiter = TokenBucketRateLimiter(
    rate_per_second=float(os.getenv("CLIENT_RATE_LIMIT_RPS", "10")),
    burst_capacity=float(os.getenv("CLIENT_RATE_LIMIT_BURST", "20")),
)
feedback_store = FeedbackStore()
run_store = RunReconstructionStore()
vector_store = RegulatoryVectorStore()
shared_llm_client = LLMClient()
claim_cache = ClaimCache()
publish_index_metrics(vector_store.get_freshness())

DEMO_FAULTS = {"unsupported_citation"}


@router.post("/claims", status_code=202)
async def submit_passenger_claim(
    request: Request,
    raw_payload: Dict[str, Any],
    prompt_version: str = Query("v2.0", description="Prompt variant version (L2)"),
    demo_fault: Optional[str] = Query(None, description="Demo item 6: 'unsupported_citation' injects a citation "
                                                        "the agent never retrieved, to show self-verification catching it"),
):
    """
    Ingestion Endpoint (F1, F2, D1, A4).
    Enforces rate limits, input guardrails, and schema/cross-field validation.
    Quarantines rejected payloads and enqueues valid claims asynchronously.
    """
    client_ip = request.client.host if request.client else "unknown"
    client_id = request.headers.get("X-Client-ID") or client_ip  # per-client budget (D7)
    correlation_id = request.headers.get("X-Correlation-ID", f"corr-{uuid.uuid4().hex[:8]}")

    # 1. Rate Limiting (D7)
    allowed, tokens = rate_limiter.is_allowed(client_id)
    if not allowed:
        CLAIM_SUBMISSIONS_TOTAL.labels(status="rate_limited").inc()
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Client loop protection active (D7).",
        )

    # 2. Input Guardrails: Prompt Injection and Toxicity Inspection (L4)
    free_text = raw_payload.get("free_text_claim_narrative", "")
    is_safe, violation, details = InputGuardrails.inspect_text(free_text)
    if not is_safe:
        quarantine_id = f"quar-{uuid.uuid4().hex[:8]}"
        quarantine_store.save_rejected_claim(
            quarantine_id=quarantine_id,
            correlation_id=correlation_id,
            raw_payload=raw_payload,
            validation_errors=[{"field": "free_text_claim_narrative", "violation": violation, "msg": details}],
            rejection_reason=f"Security guardrail failure: {violation}",
            client_ip=client_ip,
        )
        CLAIM_SUBMISSIONS_TOTAL.labels(status="quarantined").inc()
        logger.warning(f"[GUARDRAIL TRIGGERED] {violation}: {details}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Security Guardrail Violation",
                "violation_type": violation,
                "quarantine_id": quarantine_id,
                "correlation_id": correlation_id,
                "message": details,
            },
        )

    # 3. Schema & Cross-Field Validation (F2, D1)
    try:
        claim_request = PassengerClaimRequest.model_validate(raw_payload)
    except ValidationError as e:
        quarantine_id = f"quar-{uuid.uuid4().hex[:8]}"
        field_errors = []
        for err in e.errors():
            field_errors.append({
                "loc": err["loc"],
                "msg": err["msg"],
                "type": err["type"],
            })

        quarantine_store.save_rejected_claim(
            quarantine_id=quarantine_id,
            correlation_id=correlation_id,
            raw_payload=raw_payload,
            validation_errors=field_errors,
            rejection_reason="Schema or cross-field validation failure at ingestion boundary",
            client_ip=client_ip,
        )
        CLAIM_SUBMISSIONS_TOTAL.labels(status="quarantined").inc()
        logger.warning(f"[REJECT QUARANTINED] id={quarantine_id} errors={len(field_errors)}")
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Validation Failure - Record Quarantined",
                "quarantine_id": quarantine_id,
                "correlation_id": correlation_id,
                "validation_errors": field_errors,
                "rejection_reason": "Malformed request prevented from starting agent run (D1).",
            },
        )

    # 4. Asynchronous Execution Enqueue (A4)
    if prompt_version not in ("v1", "v1.0", "v2", "v2.0"):
        raise HTTPException(status_code=400, detail=f"Unknown prompt_version '{prompt_version}'")
    if demo_fault and demo_fault not in DEMO_FAULTS:
        raise HTTPException(status_code=400, detail=f"Unknown demo_fault '{demo_fault}'")
    run_id = f"run-{uuid.uuid4().hex[:12]}"
    submitted_at = time.time()

    with span("api.submit_claim", {"run_id": run_id, "correlation_id": correlation_id,
                                   "disruption_type": claim_request.disruption_type.value}):
        task_data = {
            "run_id": run_id,
            "correlation_id": correlation_id,
            "claim_data": claim_request.model_dump(mode="json"),
            "prompt_version": prompt_version,
            "submitted_at": submitted_at,
            "demo_fault": demo_fault,
            "trace_context": inject_context(),
        }

        if not task_queue.is_using_redis():
            # No Redis (local run without Docker): execute both agents inline
            orch = MultiAgentTriageOrchestrator(vector_store=vector_store, llm_client=shared_llm_client)
            result = orch.execute_claim_triage(claim_request, correlation_id=correlation_id,
                                               prompt_version=prompt_version, run_id=run_id, demo_fault=demo_fault)
            CLAIM_RUN_LATENCY.observe(time.time() - submitted_at)
            run_store.save_run(run_id, {
                "status": "COMPLETED",
                "result": result.model_dump(mode="json"),
                "worker_id": "inline-worker",
                "cached": False,
                "completed_at": time.time(),
            })
        else:
            run_store.save_run(run_id, {
                "run_id": run_id,
                "correlation_id": correlation_id,
                "status": "QUEUED",
                "submitted_at": submitted_at,
            })
            task_queue.enqueue(task_data)

    CLAIM_SUBMISSIONS_TOTAL.labels(status="accepted").inc()

    return {
        "run_id": run_id,
        "correlation_id": correlation_id,
        "status": "QUEUED" if task_queue.is_using_redis() else "COMPLETED",
        "poll_url": f"/api/v1/claims/{run_id}",
        "message": "Claim accepted. Polling available at poll_url (A4).",
    }


@router.get("/claims/{run_id}")
async def get_claim_run_status(run_id: str):
    """Poll run status and full triage reconstruction (A4, D5)."""
    run_data = run_store.get_run(run_id)
    if not run_data:
        raise HTTPException(status_code=404, detail=f"Run ID '{run_id}' not found.")
    return run_data


@router.get("/runs")
async def list_recent_runs(limit: int = 20):
    """Most recent runs (D5)."""
    return run_store.list_recent_runs(limit=limit)


@router.get("/index/freshness")
async def get_index_freshness():
    """Data freshness of the vector index: build time, source versions, chunk count, throughput (D2)."""
    return vector_store.get_freshness()


@router.get("/contract")
async def get_data_contract():
    """Published Data Contract Schema (D1)."""
    return {
        "version": "1.0.0",
        "title": "Air Passenger Compensation Claim Ingestion Contract",
        "request_schema": PassengerClaimRequest.model_json_schema(),
    }


@router.get("/quarantine")
async def list_quarantined_claims(limit: int = 50):
    """Retrieve quarantined invalid claims (F2, D1)."""
    return quarantine_store.list_records(limit=limit)


@router.get("/quarantine/{quarantine_id}")
async def get_quarantined_claim(quarantine_id: str):
    record = quarantine_store.get_record(quarantine_id)
    if not record:
        raise HTTPException(status_code=404, detail="Quarantine record not found.")
    return record


@router.post("/feedback")
async def submit_feedback(feedback: FeedbackSubmission):
    """Analyst satisfaction feedback endpoint (L6)."""
    record = feedback_store.record_feedback(feedback)
    FEEDBACK_SATISFACTION_GAUGE.set(feedback_store.get_summary()["satisfaction_rate"])
    return record


@router.get("/feedback/summary")
async def get_feedback_summary():
    """Aggregated satisfaction metrics (L6)."""
    return feedback_store.get_summary()


@router.get("/circuit-breaker")
async def get_circuit_breaker_status():
    """Circuit breaker operational state (A5)."""
    return shared_llm_client.circuit_breaker.get_status()


@router.post("/circuit-breaker/trip")
async def trip_circuit_breaker():
    """Demonstrate circuit breaker opening (Demo Item 8)."""
    shared_llm_client.circuit_breaker.trip_manually()
    return {"status": "tripped", "circuit": shared_llm_client.circuit_breaker.get_status()}


@router.post("/circuit-breaker/reset")
async def reset_circuit_breaker():
    shared_llm_client.circuit_breaker.reset_manually()
    return {"status": "reset", "circuit": shared_llm_client.circuit_breaker.get_status()}
