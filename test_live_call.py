import urllib.request
import json

payload = {
    "pnr": "6E7K9P",
    "passenger_name": "Rohan Sharma",
    "contact_email": "rohan.sharma@example.com",
    "contact_phone": "9876543210",
    "flight_number": "6E-204",
    "scheduled_departure": "2026-10-19T14:00:00Z",
    "disruption_type": "cancellation",
    "notice_hours_before_departure": 1.5,
    "block_time_hours": 2.5,
    "offered_alternate_flight": True,
    "accepted_alternate_flight": True,
    "alternate_flight_delay_hours": 4.0,
    "ticket_basic_fare_inr": 5500.0,
    "airline_fuel_charge_inr": 1000.0,
    "contact_details_provided_at_booking": True,
    "cause_attribution": "airline_operational",
    "free_text_claim_narrative": "My flight was cancelled. I was already at the airport and was told 90 minutes before departure. The airline put me on a flight four hours later, which I took. I want the compensation for cancellation."
}

req = urllib.request.Request(
    "http://127.0.0.1:8000/api/v1/claims",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode("utf-8"))
    print(f">>> Claim Submitted! Run ID: {data['run_id']}")
    poll_url = "http://127.0.0.1:8000" + data["poll_url"]

with urllib.request.urlopen(poll_url) as poll_resp:
    res = json.loads(poll_resp.read().decode("utf-8"))
    triage = res["result"]
    print("\n================ LIVE TRIAGE OUTCOME ================")
    print(f"Routing Decision      : {triage['routing_decision'].upper()}")
    print(f"Monetary Compensation : INR {triage['monetary_compensation_inr']:,.0f}")
    print(f"Facilities Entitled   : {triage['facilities_entitled']}")
    print(f"Self-Verification     : {'PASSED (A3)' if triage['self_verification_passed'] else 'FAILED'}")
    print(f"\nReasoning Summary:\n{triage['reasoning_summary']}")
    print("\nGrounded Citations:")
    for c in triage["clause_citations"]:
        print(f"  • [{c['clause_id']}] {c['document_title']}")
        print(f"    Quote: \"{c['quote']}\"")
        print(f"    Relevance: {c['relevance']}\n")
