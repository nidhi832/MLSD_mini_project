# Operational Runbook (D8)

Commands run from the project folder. Logs: `docker compose logs <service> --tail=50`.

---

## Failure mode 1: LLM provider unavailable (circuit breaker open)

**Symptom**
- Runs complete but are marked `"degraded_mode": true` with `degraded_reasons` such as
  `planner model unavailable (CircuitBreakerOpenException)`; summaries end with "[Degraded mode] …".
- Grafana: *Retry Rate* spikes (`llm_call`), then drops to zero as calls fail fast.
- Worker logs: `planner LLM call failed, using policy`.

**Confirm**
```bash
curl http://localhost:8000/api/v1/circuit-breaker          # "state": "OPEN"
docker compose ps llm-stub                                  # stub stopped? (default provider)
docker compose logs agent-worker --tail=30
```
For Groq/Gemini: check the key in `.env`, the model id, and the provider's status / free-tier limits (HTTP 429).

**Recover**
1. Restore the provider: `docker compose start llm-stub`, or fix `.env` and `docker compose up -d`.
2. Wait `CIRCUIT_BREAKER_RECOVERY_SECONDS` (30 s): the breaker half-opens and closes on the next success,
   or reset it: `curl -X POST http://localhost:8000/api/v1/circuit-breaker/reset`.
3. Optional: resubmit runs that were degraded if a model-chosen plan is required.

---

## Failure mode 2: Runs stuck in QUEUED / CLASSIFYING / PROCESSING

**Symptom**
- The web client keeps waiting; `GET /api/v1/claims/<run_id>` never reaches `COMPLETED`.

**Confirm**
```bash
docker compose ps                                   # are classifier-agent and agent-worker Up (healthy)?
docker compose exec redis redis-cli llen claim_processing_queue
docker compose exec redis redis-cli llen claim_reasoning_queue
docker compose logs classifier-agent agent-worker --tail=50
```
- `QUEUED` + growing `claim_processing_queue` → classifier-agent down.
- `QUEUED_FOR_REASONING` + growing `claim_reasoning_queue` → agent-worker down or overloaded.
- `PROCESSING` that never ends → the worker died mid-run (task is lost).

**Recover**
1. `docker compose up -d` (restarts stopped services; `restart: always` normally does this).
2. Overloaded: `docker compose up -d --scale agent-worker=3`.
3. Lost runs: resubmit the claim (identical claims are idempotent: same facts → same determination).

---

## Failure mode 3: Retrieval / index problems (unexpected escalations)

**Symptom**
- Many runs escalate with "Self-verification failed (A3): citation … does not resolve to retrieved text".
- Grafana *Self-Verification Outcomes*: `failed_escalated` rising.

**Confirm**
```bash
curl http://localhost:8000/api/v1/index/freshness   # total_chunks should be 52; check embedding_backend
docker compose exec api python -m pytest tests/test_agent_graph.py -q
```

**Recover**
```bash
docker compose exec api python -m src.ingestion.pipeline --force
docker compose restart agent-worker
```
If `embedding_backend` is `builtin-hashing-384`, the MiniLM model was not downloaded during the build;
rebuild with network access: `docker compose build --no-cache api`.

---

## Failure mode 4: Docker Desktop crashes or the build fails

**Symptom**: `rpc error … EOF`, `Docker Desktop is unable to start`, or downloads failing with `EOF`.

**Recover**
1. Quit Docker Desktop, run `wsl --shutdown`, reopen it; restart the PC if it still fails.
2. Check free disk space on the drive holding Docker's disk image (≥ 15 GB).
3. Pull base images one at a time (`docker pull redis:7.2-alpine`, etc.) on an unstable network, then
   `docker compose up --build -d` again; finished layers are cached.
4. Keep the CPU-only PyTorch line in the Dockerfile (see INC-2026-11-10-01).
