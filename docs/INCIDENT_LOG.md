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
