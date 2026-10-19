"""
Agent 1: Claim Intake, Classification & Entity Extraction Agent (F3, A2).
Extracts operational parameters and forms the structured handoff to Agent 2.
"""

from typing import Dict, Any
from src.core.contracts import PassengerClaimRequest, DisruptionType
from src.agents.llm_client import LLMClient


class ClaimClassifierAgent:
    def __init__(self, llm_client: LLMClient, system_prompt: str):
        self.llm_client = llm_client
        self.system_prompt = system_prompt

    def process(self, claim: PassengerClaimRequest) -> Dict[str, Any]:
        """
        Executes classification and extraction.
        Produces structured handoff payload for Agent 2.
        """
        # Execute prompt step with LLM backend
        user_prompt = (
            f"Analyze claim narrative:\n"
            f"Passenger: {claim.passenger_name}, Flight: {claim.flight_number}, Route Carrier: {claim.operating_carrier}\n"
            f"Narrative: \"{claim.free_text_claim_narrative}\"\n"
            f"Pre-parsed Metadata: Disruption={claim.disruption_type.value}, BlockTime={claim.block_time_hours}h, "
            f"Notice={claim.notice_hours_before_departure}h, AlternateOffered={claim.offered_alternate_flight}, "
            f"AlternateAccepted={claim.accepted_alternate_flight}, CauseAttribution={claim.cause_attribution}."
        )

        llm_output = self.llm_client.generate(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            step_name="agent_1_classification_extraction",
        )

        # Structured handoff payload
        handoff_payload = {
            "disruption_type": claim.disruption_type,
            "flight_number": claim.flight_number,
            "operating_carrier": claim.operating_carrier,
            "block_time_hours": claim.block_time_hours,
            "notice_hours_before_departure": claim.notice_hours_before_departure,
            "delay_hours": claim.delay_hours or 0.0,
            "offered_alternate_flight": claim.offered_alternate_flight,
            "accepted_alternate_flight": claim.accepted_alternate_flight,
            "alternate_flight_delay_hours": claim.alternate_flight_delay_hours or 0.0,
            "contact_details_provided_at_booking": claim.contact_details_provided_at_booking,
            "ticket_basic_fare_inr": claim.ticket_basic_fare_inr,
            "airline_fuel_charge_inr": claim.airline_fuel_charge_inr,
            "cause_attribution": claim.cause_attribution,
            "free_text_claim_narrative": claim.free_text_claim_narrative,
            "llm_classification_summary": llm_output,
        }

        return handoff_payload
