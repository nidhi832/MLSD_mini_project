# Operational Runbook: Air Passenger Claim Triage Service

This runbook covers the two most critical failure modes in accordance with Requirement **D8**.

---

## 🚨 Failure Mode 1: LLM Provider Outage / Circuit Breaker Trip

### 1. Symptoms
- Ingestion accepts requests (`202 Accepted`), but worker logs report `CircuitBreakerOpenException`.
- Dashboard shows an increase in `circuit_breaker_trips_total`.
- Triage results automatically route to `ESCALATE` with reasoning summary stating *"Graceful degradation triggered mid-run... Escalated to human operator"*.
- System latency drops to near-zero as provider calls fail fast without hanging.

### 2. Confirmation
1. Query the circuit breaker status endpoint:
   ```bash
   curl -s http://localhost:8000/api/v1/circuit-breaker
   ```
   **Expected Output when tripped:**
   ```json
   {
     "name": "llm_provider",
     "state": "OPEN",
     "failure_count": 3,
     "threshold": 3,
     "total_trips": 1
   }
   ```
2. Check worker logs:
   ```bash
   docker compose logs agent-worker --tail=50
   ```
   Look for `Circuit breaker 'llm_provider' is OPEN`.

### 3. Recovery Procedure
1. Verify provider connectivity:
   - If using cloud provider (OpenAI/Anthropic), verify endpoint status and API quota.
   - If using local stub/Ollama, ensure process is healthy.
2. Once provider connectivity is restored, reset the circuit breaker via REST:
   ```bash
   curl -X POST http://localhost:8000/api/v1/circuit-breaker/reset
   ```
3. Alternatively, allow the built-in automatic half-open probe (recovers after 30 seconds).

---

## 🚨 Failure Mode 2: Vector Index Desynchronization or Missing Corpus Chunks

### 1. Symptoms
- Agent 2 self-verification fails consistently (`self_verification_passed: false`).
- Valid regulatory clauses fail to resolve (`could not be resolved directly to retrieved regulatory text`).
- Claims with clear entitlements are incorrectly escalated to human review.

### 2. Confirmation
1. Inspect data freshness and vector store manifest:
   ```bash
   python -c "from src.ingestion.vector_store import RegulatoryVectorStore; print(RegulatoryVectorStore().get_freshness())"
   ```
2. Verify chunk count:
   - Expected: 52 chunks (36 CAR chunks + 16 Airline Charter chunks).
   - If `total_chunks < 50`, the index is incomplete or corrupt.

### 3. Recovery Procedure
1. Execute the idempotent corpus ingestion CLI with `--force`:
   ```bash
   python -m src.ingestion.pipeline --force
   ```
2. Verify output confirms 52 chunks indexed:
   ```text
   Total Chunks: 52
   Newly Embedded: 52
   Throughput: > 50 chunks/sec
   ```
3. Restart worker instances to reload memory vectors:
   ```bash
   docker compose restart agent-worker
   ```
