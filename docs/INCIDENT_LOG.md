# Incident Log (D8)

This incident log tracks genuine engineering failures and root cause analyses during system development.

---

## 📌 INC-2026-10-19-01: Ingestion Boundary Permitted Inconsistent Alternate Flight State Leading to Erroneous Payout Recommendation

- **Date / Time**: 2026-10-19 21:42:00 IST
- **Severity**: High (Financial Risk / Regulatory Violation)
- **Component**: `src/core/contracts.py` & Ingestion Boundary
- **Impact**: Test payload with `accepted_alternate_flight = True` but `offered_alternate_flight = False` bypassed ingestion, causing inconsistent downstream reasoning and ambiguous CAR 3.3.2 triage.

### Description
During initial testing of edge cases, a payload was submitted where `accepted_alternate_flight` was marked `True`, but `offered_alternate_flight` was set to `False`. Logically, a passenger cannot accept and board an alternate flight that was never offered by the airline.
Because field validation previously only inspected individual field ranges and types without cross-field correlation rules, the record was accepted with HTTP 200/202 instead of being quarantined.

### Root Cause
Missing cross-field validation rule at the Pydantic ingestion schema boundary (Requirement F2, D1).

### Corrective Action Taken
1. Added Pydantic `@model_validator(mode="after")` inside `PassengerClaimRequest` in `src/core/contracts.py`:
   ```python
   if self.accepted_alternate_flight and not self.offered_alternate_flight:
       raise ValueError("Cross-field violation: 'accepted_alternate_flight' cannot be true when 'offered_alternate_flight' is false.")
   ```
2. Connected validation failures directly to `QuarantineStore` (`src/core/quarantine.py`), which immediately persists the payload and returns HTTP 422 with quarantine tracking ID.
3. Added automated unit test `test_cross_field_validation_rejection` in `tests/test_claim_system.py` to permanently guard against regressions.
- **Status**: Resolved & Verified in CI.

---

## INC-2026-11-10-01: Docker Desktop crashed while building the images
- **Symptom**: `docker compose up --build` failed with `rpc error: ... EOF`, then `Docker Desktop is unable to start`.
- **Root cause**: `pip install sentence-transformers` pulled the CUDA (GPU) build of PyTorch, several GB per image; two images (api, agent-worker) were exported at the same time and exhausted Docker Desktop's ~5.7 GB memory.
- **Fix**: install CPU-only PyTorch first in the Dockerfile; build one shared image used by all app services. Export time fell from ~376 s to ~59 s.

## INC-2026-11-10-02: Claims stayed QUEUED forever under Docker Compose
- **Symptom**: the web client polled `/api/v1/claims/<run_id>` and never got a result.
- **Root cause**: the worker saved run results to its own container's `data/runs/`; the API container read its own (empty) folder. Worked in local inline mode, broke only with separate containers.
- **Fix**: run records, the whole-run cache and circuit-breaker state moved to Redis (`src/core/shared_state.py`). Verified end-to-end with API + two workers + Redis.

## INC-2026-11-10-03: Clause lookup returned the wrong clause, hidden by weak verification
- **Symptom**: `lookup_clause_by_id("CAR-3.3.2")` returned the bare `CAR-3.3` heading; verification still passed.
- **Root cause**: substring matching on clause ids; verification accepted a citation if any 1 of 4 quote words appeared in retrieved text, and quotes were hard-coded rather than taken from retrieved passages.
- **Fix**: exact clause lookup (with sub-clauses merged); citations now quote retrieved lines verbatim and verification requires the exact line. Covered by `tests/test_agent_graph.py`.
