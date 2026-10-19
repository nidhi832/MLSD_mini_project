"""
Locust Load Testing Suite (D6, A7).
Tests end-to-end processing latency against declared SLO (p95 <= 5.0s)
using LLM stub mode to measure system architecture, queuing, and worker throughput.
"""

import time
from locust import HttpUser, task, between


class PassengerClaimLoadUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(4)
    def submit_and_poll_standard_claim(self):
        """End-to-end claim submission and poll until completed (D6)."""
        start_time = time.time()

        payload = {
            "pnr": "6E8K2L",
            "passenger_name": "Karthik Raja",
            "contact_email": "karthik@example.com",
            "contact_phone": "9876543210",
            "contact_details_provided_at_booking": True,
            "flight_number": "6E-305",
            "scheduled_departure": "2026-10-19T14:00:00Z",
            "disruption_type": "cancellation",
            "notice_hours_before_departure": 1.5,
            "block_time_hours": 2.0,
            "offered_alternate_flight": True,
            "accepted_alternate_flight": True,
            "alternate_flight_delay_hours": 4.0,
            "ticket_basic_fare_inr": 5000.0,
            "airline_fuel_charge_inr": 1000.0,
            "cause_attribution": "airline_operational",
            "free_text_claim_narrative": "My flight was cancelled at airport. I took alternate flight 4 hours later. Section 4.1 scenario.",
        }

        # 1. Submit claim
        with self.client.post("/api/v1/claims", json=payload, catch_response=True) as resp:
            if resp.status_code != 202:
                resp.failure(f"Submission failed with status {resp.status_code}")
                return
            data = resp.json()
            run_id = data.get("run_id")

        # 2. Poll until completed (measuring full run, not submission alone)
        completed = False
        attempts = 0
        while not completed and attempts < 30:
            time.sleep(0.1)
            attempts += 1
            with self.client.get(f"/api/v1/claims/{run_id}", catch_response=True) as poll_resp:
                if poll_resp.status_code == 200:
                    status_data = poll_resp.json()
                    if status_data.get("status") == "COMPLETED":
                        completed = True
                        break
                    elif status_data.get("status") == "FAILED":
                        poll_resp.failure("Run status marked as FAILED")
                        break

        total_duration = time.time() - start_time
        # Record end-to-end custom event
        self.environment.events.request.fire(
            request_type="E2E_RUN",
            name="Claim_Submission_To_Completion",
            response_time=total_duration * 1000,
            response_length=0,
            exception=None if completed else TimeoutError("Poll timeout"),
            context={},
        )

    @task(1)
    def submit_malformed_claim(self):
        """Tests ingestion boundary quarantine under load (D1, F2)."""
        malformed_payload = {
            "pnr": "INVALID",
            "passenger_name": "X",
            "free_text_claim_narrative": "Too short",
        }
        with self.client.post("/api/v1/claims", json=malformed_payload, catch_response=True) as resp:
            if resp.status_code == 422:
                resp.success()
            else:
                resp.failure(f"Expected 422 quarantine reject, got {resp.status_code}")
