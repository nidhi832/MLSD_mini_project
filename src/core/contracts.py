"""
Data contracts for Air Passenger Compensation Claim Triage Service.
Enforces schema, type, range, categorical, and cross-field validation (F1, F2, D1).
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime
import re
from pydantic import BaseModel, Field, model_validator, field_validator


class DisruptionType(str, Enum):
    CANCELLATION = "cancellation"
    DENIED_BOARDING = "denied_boarding"
    DELAY = "delay"
    DOWNGRADING = "downgrading"


class RoutingDecision(str, Enum):
    PAY_AS_COMPUTED = "pay-as-computed"
    REQUEST_FURTHER_EVIDENCE = "request-further-evidence"
    DECLINE = "decline"
    ESCALATE = "escalate"


class PassengerClaimRequest(BaseModel):
    """External Data Contract for Passenger Claim Submission (D1, F1, F2)."""

    claim_id: Optional[str] = Field(
        default=None,
        description="Optional client-side tracking ID. Auto-generated if omitted.",
    )
    pnr: str = Field(
        ...,
        pattern=r"^[A-Z0-9]{6}$",
        description="6-character alphanumeric Passenger Name Record (PNR).",
        examples=["6E7K9P"],
    )
    passenger_name: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Full name of the passenger.",
        examples=["Rohan Sharma"],
    )
    contact_email: str = Field(
        ...,
        description="Valid email address for notification.",
        examples=["rohan.sharma@example.com"],
    )
    contact_phone: str = Field(
        ...,
        pattern=r"^\+?[0-9]{10,14}$",
        description="10 to 14 digit contact phone number.",
        examples=["9876543210"],
    )
    contact_details_provided_at_booking: bool = Field(
        ...,
        description="Whether correct contact details were furnished at time of booking (CAR 3.3.4b).",
    )
    flight_number: str = Field(
        ...,
        pattern=r"^[A-Z0-9]{2,3}\-?[0-9]{3,4}$",
        description="IATA/ICAO Flight number (e.g. 6E-204, AI-101).",
        examples=["6E-204"],
    )
    operating_carrier: str = Field(
        default="IndiGo",
        description="Operating airline carrier name.",
        examples=["IndiGo"],
    )
    scheduled_departure: datetime = Field(
        ...,
        description="Original scheduled flight departure date and time.",
    )
    disruption_type: DisruptionType = Field(
        ...,
        description="Category of disruption encountered.",
    )
    notice_hours_before_departure: float = Field(
        ...,
        ge=0.0,
        le=720.0,
        description="Notice provided before departure in hours (0.0 if informed at airport).",
        examples=[1.5],
    )
    block_time_hours: float = Field(
        ...,
        gt=0.0,
        le=20.0,
        description="Scheduled block time of the flight in hours (CAR 2.1).",
        examples=[1.5],
    )
    delay_hours: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Hours of arrival delay experienced (mandatory if disruption is delay).",
        examples=[4.0],
    )
    offered_alternate_flight: bool = Field(
        ...,
        description="Whether an alternate flight was offered by the carrier.",
    )
    accepted_alternate_flight: bool = Field(
        ...,
        description="Whether the passenger accepted and traveled on the alternate flight.",
    )
    alternate_flight_delay_hours: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Delay in departure time of alternate flight compared to original flight.",
        examples=[4.0],
    )
    ticket_basic_fare_inr: float = Field(
        ...,
        gt=0.0,
        description="Booked one-way basic fare in INR (CAR 3.3.2).",
        examples=[4500.0],
    )
    airline_fuel_charge_inr: float = Field(
        default=0.0,
        ge=0.0,
        description="Airline fuel charge (YQ/YR) in INR.",
        examples=[1000.0],
    )
    cause_attribution: str = Field(
        default="airline_operational",
        description="Attributed cause of disruption (e.g., airline_operational, weather, atc, disputed).",
        examples=["airline_operational"],
    )
    free_text_claim_narrative: str = Field(
        ...,
        min_length=10,
        max_length=3000,
        description="Passenger's original narrative describing the claim.",
        examples=[
            "My flight was cancelled. I was already at the airport and was told 90 minutes before departure. The airline put me on a flight four hours later, which I took. I want the compensation for cancellation."
        ],
    )

    @field_validator("contact_email")
    @classmethod
    def validate_email_format(cls, v: str) -> str:
        pattern = r"^[\w\.\+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-\.]+$"
        if not re.match(pattern, v):
            raise ValueError(f"Invalid email address format: {v}")
        return v

    @model_validator(mode="after")
    def validate_cross_field_rules(self) -> "PassengerClaimRequest":
        errors = []

        # Rule 1: Cannot accept an alternate flight if one was never offered!
        if self.accepted_alternate_flight and not self.offered_alternate_flight:
            errors.append(
                "Cross-field violation: 'accepted_alternate_flight' cannot be true when 'offered_alternate_flight' is false."
            )

        # Rule 2: If disruption_type is DELAY, delay_hours must be provided and > 0
        if self.disruption_type == DisruptionType.DELAY:
            if self.delay_hours is None or self.delay_hours <= 0.0:
                errors.append(
                    "Cross-field violation: 'delay_hours' must be provided and greater than 0 for disruption_type 'delay'."
                )

        # Rule 3: If alternate flight accepted, alternate_flight_delay_hours should be positive
        if self.accepted_alternate_flight and (self.alternate_flight_delay_hours is None or self.alternate_flight_delay_hours < 0):
            errors.append(
                "Cross-field violation: 'alternate_flight_delay_hours' must be provided when alternate flight was accepted."
            )

        if errors:
            raise ValueError("; ".join(errors))

        return self


class Citation(BaseModel):
    clause_id: str = Field(..., description="Legal clause or document identifier (e.g. CAR 3.3.2).")
    document_title: str = Field(..., description="Corpus document title.")
    quote: str = Field(..., description="Exact quoted provision from the retrieved text.")
    relevance: str = Field(..., description="Explanation of how this clause applies.")


class EntitlementEvaluationResult(BaseModel):
    """Complete structured output from the multi-agent reasoning layer (F5, F6, F7, A3)."""

    run_id: str
    correlation_id: str
    claim_id: Optional[str] = None
    pnr: str
    disruption_classification: DisruptionType
    extracted_entities: Dict[str, Any]
    routing_decision: RoutingDecision
    monetary_compensation_inr: float = Field(
        default=0.0,
        description="Calculated financial compensation payable (if any).",
    )
    facilities_entitled: List[str] = Field(
        default_factory=list,
        description="Statutory non-monetary facilities entitled (e.g. meals, hotel, transport).",
    )
    refund_entitled: bool = Field(
        default=False,
        description="Whether full or partial refund of ticket is entitled.",
    )
    clause_citations: List[Citation] = Field(
        default_factory=list,
        description="Grounded clause-level citations supporting the determination.",
    )
    reasoning_summary: str = Field(
        ...,
        description="Concise legal and factual basis for the triage recommendation.",
    )
    escalation_reason: Optional[str] = Field(
        default=None,
        description="Explanation if routed to escalation (F7).",
    )
    prompt_version: str = Field(
        default="v2.0",
        description="Version identifier of prompt used (L2).",
    )
    self_verification_passed: bool = Field(
        default=False,
        description="Whether pre-response verification against retrieved text passed (A3).",
    )
    steps_executed: int = Field(default=0)
    tool_calls_made: int = Field(default=0)
    tokens_prompt: int = Field(default=0)
    tokens_completion: int = Field(default=0)
    cost_usd: float = Field(default=0.0)
    execution_time_seconds: float = Field(default=0.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
