"""
Locust load test (D6, A7). Measures a complete run - submission to final result -
against the declared SLO (p95 <= 5.0 s), with the LLM replaced by the llm-stub
service (LLM_PROVIDER=stub-service), so the test measures this system and not a
provider's rate limits.

Run inside the API container (nothing to install on your laptop):
    docker compose exec api locust -f locust/locustfile.py --headless -u 10 -r 2 --run-time 60s \
        -H http://localhost:8000 --csv data/runs/locust_u10

Read the "E2E_RUN Claim_Submission_To_Completion" row: that is the end-to-end run.
Repeat with -u 5, 10, 20, 40 to find the concurrency at which p95 exceeds 5 s, then
`docker compose up -d --scale agent-worker=3` and repeat to measure the effect of scaling.

Every simulated user is its own client (X-Client-ID) with a unique PNR per claim,
so the per-client rate limit and the whole-run cache do not distort the measurement.
Set LOCUST_ALLOW_CACHE=1 to resubmit identical claims and measure the cache effect (D7).
"""

import json
import os
import random
import string
import time
import uuid

from locust import HttpUser, between, task

SCENARIOS = json.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "evaluation_set_20.json"),
                           encoding="utf-8"))
VALID = [s["claim_payload"] for s in SCENARIOS]
ALLOW_CACHE = os.getenv("LOCUST_ALLOW_CACHE", "0") == "1"
POLL_TIMEOUT_S = float(os.getenv("LOCUST_POLL_TIMEOUT_SECONDS", "60"))


def _pnr() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


class PassengerClaimLoadUser(HttpUser):
    wait_time = between(0.5, 1.5)

    def on_start(self):
        self.client_id = f"locust-{uuid.uuid4().hex[:8]}"

    @task(5)
    def submit_and_poll(self):
        payload = dict(random.choice(VALID))
        if not ALLOW_CACHE:
            payload["pnr"] = _pnr()
        headers = {"X-Client-ID": self.client_id}
        start = time.time()

        with self.client.post("/api/v1/claims", json=payload, headers=headers, catch_response=True,
                              name="POST /api/v1/claims") as resp:
            if resp.status_code != 202:
                resp.failure(f"submission returned {resp.status_code}")
                return
            run_id = resp.json()["run_id"]

        status, error = None, None
        while time.time() - start < POLL_TIMEOUT_S:
            time.sleep(0.2)
            with self.client.get(f"/api/v1/claims/{run_id}", catch_response=True,
                                 name="GET /api/v1/claims/[run_id]") as poll:
                if poll.status_code != 200:
                    poll.failure(f"poll returned {poll.status_code}")
                    continue
                status = poll.json().get("status")
                if status in ("COMPLETED", "FAILED"):
                    break
        if status != "COMPLETED":
            error = RuntimeError(f"run ended as {status or 'TIMEOUT'}")

        self.environment.events.request.fire(
            request_type="E2E_RUN",
            name="Claim_Submission_To_Completion",
            response_time=(time.time() - start) * 1000,
            response_length=0,
            exception=error,
            context={},
        )

    @task(1)
    def submit_malformed_claim(self):
        """Malformed claims must be quarantined at the boundary (D1) under load too."""
        with self.client.post("/api/v1/claims", json={"pnr": "INVALID", "passenger_name": "X",
                                                      "free_text_claim_narrative": "Too short"},
                              headers={"X-Client-ID": self.client_id}, catch_response=True,
                              name="POST /api/v1/claims (malformed)") as resp:
            if resp.status_code == 422:
                resp.success()
            else:
                resp.failure(f"expected 422, got {resp.status_code}")
