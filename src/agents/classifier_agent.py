"""
Agent 1: Claim Intake, Classification & Entity Extraction Agent (F3, A2).

Responsibility: turn a validated claim into the structured hand-off that Agent 2
reasons over, and check that the passenger's own narrative agrees with the
structured fields. It runs as its own service (classifier-agent) and hands off
to Agent 2 through the Redis queue `claim_reasoning_queue`.

With a real LLM the model extracts the disruption category, notice, block time,
alternate-flight and contact-detail entities from the free text (F3). Where the
narrative contradicts the form (e.g. the form says the alternate flight was not
taken but the passenger writes "which I took"), the contradiction is recorded in
`fact_conflicts` and Agent 2 escalates instead of guessing (F7). In stub mode
(load tests) a keyword extractor does the same job without a model call.
"""

import re
from typing import Any, Dict, List, Optional

from src.agents.llm_client import LLMClient
from src.core.contracts import DisruptionType, PassengerClaimRequest
from src.core.observability import logger

_KEYWORDS = {
    DisruptionType.CANCELLATION: [r"\bcancel"],
    DisruptionType.DELAY: [r"\bdelay", r"\blate\b"],
    DisruptionType.DENIED_BOARDING: [r"denied boarding", r"\bdenied\b", r"offloaded", r"overbook",
                                     r"not allowed to board", r"refused boarding", r"bumped"],
    DisruptionType.DOWNGRADING: [r"downgrad", r"lower class", r"economy instead"],
}


class ClaimClassifierAgent:
    def __init__(self, llm_client: LLMClient, system_prompt: str):
        self.llm_client = llm_client
        self.system_prompt = system_prompt

    # ------------------------------------------------------------------ extraction
    def _keyword_extract(self, narrative: str) -> Dict[str, Any]:
        text = narrative.lower()
        found = [d.value for d, kws in _KEYWORDS.items() if any(re.search(k, text) for k in kws)]
        took_alt: Optional[bool] = None
        if re.search(r"\b(which|that) i (took|boarded|accepted)\b|\bi (took|boarded|accepted) (it|the (alternate|other|new|later) flight)", text):
            took_alt = True
        elif re.search(r"\bi (refused|declined|did not take|didn't take|rejected)\b", text):
            took_alt = False
        return {"disruption_types_mentioned": found, "took_alternate_flight": took_alt, "method": "keyword"}

    def _llm_extract(self, claim: PassengerClaimRequest) -> Optional[Dict[str, Any]]:
        prompt = (
            f'Passenger narrative: "{claim.free_text_claim_narrative}"\n\n'
            "Extract ONLY what the narrative itself states. Reply with JSON:\n"
            '{"disruption_type": "cancellation|delay|denied_boarding|downgrading|unclear",'
            ' "notice_hours_before_departure": number or null,'
            ' "took_alternate_flight": true|false|null,'
            ' "alternate_flight_delay_hours": number or null,'
            ' "cause_mentioned": string or null}'
        )
        data = self.llm_client.generate_json(self.system_prompt, prompt, step_name="agent_1_classification_extraction")
        if not data:
            return None
        data["method"] = "llm"
        return data

    @staticmethod
    def _conflicts(claim: PassengerClaimRequest, extracted: Dict[str, Any]) -> List[str]:
        conflicts = []
        dtype = claim.disruption_type.value
        if extracted.get("method") == "llm":
            said = str(extracted.get("disruption_type") or "unclear").lower()
            if said in {d.value for d in DisruptionType} and said != dtype:
                conflicts.append(f"narrative describes '{said}' but disruption_type is '{dtype}'")
        else:
            mentioned = extracted.get("disruption_types_mentioned") or []
            if mentioned and dtype not in mentioned and len(mentioned) == 1:
                conflicts.append(f"narrative describes '{mentioned[0]}' but disruption_type is '{dtype}'")
        took = extracted.get("took_alternate_flight")
        if took is True and not claim.accepted_alternate_flight:
            conflicts.append("narrative says the alternate flight was taken but accepted_alternate_flight is false")
        if took is False and claim.accepted_alternate_flight:
            conflicts.append("narrative says the alternate flight was refused but accepted_alternate_flight is true")
        return conflicts

    # ------------------------------------------------------------------ main
    def process(self, claim: PassengerClaimRequest) -> Dict[str, Any]:
        extracted: Optional[Dict[str, Any]] = None
        degraded = False
        if not self.llm_client.is_stub:
            try:
                extracted = self._llm_extract(claim)
            except Exception as e:
                # Degradation path (A5): fall back to keyword extraction rather than failing the run.
                logger.warning(f"Agent 1 LLM extraction failed, using keyword extraction: {type(e).__name__}: {e}")
                degraded = True
        else:
            # Stub call keeps token/cost accounting and the circuit breaker exercised under load tests (D6).
            try:
                self.llm_client.generate(self.system_prompt, claim.free_text_claim_narrative,
                                         step_name="agent_1_classification_extraction")
            except Exception:
                degraded = True
        if not extracted:
            extracted = self._keyword_extract(claim.free_text_claim_narrative)

        conflicts = self._conflicts(claim, extracted)

        return {
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
            "narrative_extraction": extracted,
            "fact_conflicts": conflicts,
            "llm_degraded": degraded,
        }
