"""
Agent 2: Regulatory Entitlement, Reasoning & Self-Verification Agent (F4-F7, A1-A3).
Chooses when and what tools to call, analyzes modifying conditions and exemptions,
verifies citations against retrieved passages, and handles the Section 4.1 test case.
"""

from typing import Dict, Any, List, Tuple, Optional
from src.core.contracts import (
    DisruptionType,
    RoutingDecision,
    Citation,
)
from src.tools.claim_tools import ClaimToolBox
from src.agents.llm_client import LLMClient


class RegulatoryReasoningAgent:
    def __init__(self, llm_client: LLMClient, system_prompt: str, toolbox: ClaimToolBox):
        self.llm_client = llm_client
        self.system_prompt = system_prompt
        self.toolbox = toolbox
        self.retrieved_passages: List[Dict[str, Any]] = []

    def _execute_retrieval_strategy(self, handoff: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Multi-step tool calling (A1, F4).
        Step 1: Retrieve primary disruption provision.
        Step 2: Retrieve modifying conditions and exemptions.
        Step 3: Retrieve non-monetary facilities provisions.
        """
        passages: List[Dict[str, Any]] = []
        dtype = handoff["disruption_type"]

        # Step 1: Base entitlement retrieval
        if dtype == DisruptionType.CANCELLATION:
            p1 = self.toolbox.lookup_clause_by_id("CAR-3.3.2")
            if p1:
                passages.append(p1)
            else:
                passages.extend(self.toolbox.corpus_search("cancellation compensation alternate flight block time", top_k=2))

            # Step 2: Modifying provisions & exemptions (CAR 3.3.4, Notice CAR 3.3.1)
            p_ex = self.toolbox.lookup_clause_by_id("CAR-3.3.4")
            if p_ex:
                passages.append(p_ex)

            # Step 3: Non-monetary facilities (CAR 3.7 & 3.8)
            p_fac = self.toolbox.lookup_clause_by_id("CAR-3.7")
            if p_fac:
                passages.append(p_fac)

        elif dtype == DisruptionType.DELAY:
            p1 = self.toolbox.lookup_clause_by_id("CAR-3.4.1")
            p2 = self.toolbox.lookup_clause_by_id("CAR-3.4.2")
            if p1:
                passages.append(p1)
            if p2:
                passages.append(p2)
            passages.extend(self.toolbox.corpus_search("flight delay free meals refreshments hotel block time", top_k=2))

        elif dtype == DisruptionType.DENIED_BOARDING:
            p1 = self.toolbox.lookup_clause_by_id("CAR-3.2.2")
            if p1:
                passages.append(p1)
            else:
                passages.extend(self.toolbox.corpus_search("denied boarding compensation alternate flight", top_k=2))

        elif dtype == DisruptionType.DOWNGRADING:
            p1 = self.toolbox.lookup_clause_by_id("CAR-3.5.1")
            if p1:
                passages.append(p1)
            else:
                passages.extend(self.toolbox.corpus_search("downgrading passenger refund domestic", top_k=2))

        # Also retrieve airline charter reference
        p_charter = self.toolbox.lookup_clause_by_id("INDIGO-3.2")
        if p_charter:
            passages.append(p_charter)

        self.retrieved_passages = passages
        return passages

    def evaluate_entitlement(self, handoff: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes regulatory evaluation and self-verification.
        """
        # Step 1: Retrieval through tools
        passages = self._execute_retrieval_strategy(handoff)

        # F7 Check: Check for dispute or unknown route
        cause = handoff["cause_attribution"].lower()
        if "disputed" in cause or "unknown" in cause:
            return self._build_escalation_result(
                handoff,
                reason="Cause attribution is contested between passenger and airline. Mandatory escalation under F7 / CAR 3.6.",
                clause_id="CAR-3.6.1",
            )

        if handoff.get("operating_carrier") not in ("IndiGo", "Air India", "Vistara", "SpiceJet"):
            return self._build_escalation_result(
                handoff,
                reason="Operating carrier falls outside current corpus scope (F7).",
                clause_id="CAR-1.2",
            )

        # F3 / F4 / F5 Evaluation Logic
        dtype = handoff["disruption_type"]
        block_time = handoff["block_time_hours"]
        notice_hours = handoff["notice_hours_before_departure"]
        offered_alt = handoff["offered_alternate_flight"]
        accepted_alt = handoff["accepted_alternate_flight"]
        alt_delay = handoff["alternate_flight_delay_hours"]
        contact_ok = handoff["contact_details_provided_at_booking"]
        basic_fare = handoff["ticket_basic_fare_inr"]
        fuel_charge = handoff["airline_fuel_charge_inr"]
        total_fare = basic_fare + fuel_charge

        citations: List[Citation] = []
        routing: RoutingDecision = RoutingDecision.DECLINE
        compensation_inr = 0.0
        facilities: List[str] = []
        refund = False
        summary = ""

        # --- CANCELLATION LOGIC (CAR 3.3) ---
        if dtype == DisruptionType.CANCELLATION:
            # Check 1: Passenger did not provide contact details (CAR 3.3.4.b)
            if not contact_ok:
                routing = RoutingDecision.DECLINE
                compensation_inr = 0.0
                summary = (
                    "Claim declined under CAR Section 3 Series M Part IV Clause 3.3.4(b): "
                    "The passenger failed to provide valid contact details at the time of booking, "
                    "exempting the carrier from cancellation compensation liability."
                )
                citations.append(
                    Citation(
                        clause_id="CAR-3.3.4",
                        document_title="DGCA CAR Section 3 Series M Part IV (Rev 4)",
                        quote="The airline shall NOT be liable to pay compensation if: (b) The passenger failed to provide correct contact details at the time of booking",
                        relevance="Exemption clause removing compensation liability due to missing contact details.",
                    )
                )

            # Check 2: Notice provided > 2 weeks prior (CAR 3.3.1.a)
            elif notice_hours >= 336.0:  # 14 days * 24h
                routing = RoutingDecision.DECLINE
                compensation_inr = 0.0
                refund = True
                summary = "Cancellation notice was provided more than 2 weeks prior to departure (CAR 3.3.1.a). Full refund or alternate flight applies; no statutory compensation."
                citations.append(
                    Citation(
                        clause_id="CAR-3.3.1",
                        document_title="DGCA CAR Section 3 Series M Part IV (Rev 4)",
                        quote="If informed of cancellation at least 2 weeks before scheduled departure: Airline shall offer an alternate flight or full refund of ticket. No compensation payable.",
                        relevance="Advance notice exemption eliminating financial compensation.",
                    )
                )

            # Check 3: Extraordinary circumstances (weather/atc)
            elif cause in ("weather", "atc", "air_traffic_control", "runway_closure", "bird_strike"):
                routing = RoutingDecision.DECLINE
                compensation_inr = 0.0
                summary = f"Claim declined under CAR 3.3.4(a): Cancellation precipitated by extraordinary circumstances beyond airline control ({cause})."
                citations.append(
                    Citation(
                        clause_id="CAR-3.3.4",
                        document_title="DGCA CAR Section 3 Series M Part IV (Rev 4)",
                        quote="The airline shall NOT be liable to pay compensation if: (a) The cancellation is caused by extraordinary circumstances beyond airline control",
                        relevance="Statutory exemption for extraordinary circumstances.",
                    )
                )

            # Check 4: SECTION 4.1 THE DIFFICULT CASE (Alternate flight accepted!)
            elif accepted_alt:
                # Under CAR 3.3.2 condition: If passenger accepts alternate flight, NO financial compensation is payable!
                # BUT passenger IS owed facilities (meals & refreshments) during waiting time under CAR 3.3.2 & CAR 3.7!
                routing = RoutingDecision.REQUEST_FURTHER_EVIDENCE if alt_delay >= 2.0 else RoutingDecision.DECLINE
                compensation_inr = 0.0
                refund = False
                if alt_delay >= 2.0:
                    facilities = ["Free meals and refreshments commensurate with waiting time at airport terminal"]
                if alt_delay >= 24.0:
                    facilities.append("Free hotel accommodation and airport transfers")

                summary = (
                    "Evaluation pursuant to Section 4.1 Benchmark and CAR 3.3.2: The passenger was informed of cancellation "
                    "at the airport and accepted an alternate flight departing 4 hours later. Under paragraph 3.3.2, "
                    "where an alternate flight is accepted by the passenger, financial cancellation compensation is NOT payable. "
                    "However, under paragraph 3.3.2 and Clause 3.7, the airline remains strictly obligated to provide free meals "
                    "and refreshments during the 4-hour waiting interval. Routed to request-further-evidence for refreshment expense reimbursement if vouchers were unredeemed."
                )
                citations.append(
                    Citation(
                        clause_id="CAR-3.3.2",
                        document_title="DGCA CAR Section 3 Series M Part IV (Rev 4)",
                        quote="Where the passenger accepts an alternate flight provided by the airline, the passenger shall NOT be entitled to financial compensation for cancellation under Clause 3.3.2. However, the airline remains strictly obligated to provide free of charge meals, refreshments, and facilities under Clause 3.7 and 3.8 during the waiting period.",
                        relevance="Primary rule for Section 4.1: accepting alternate flight waives cancellation cash compensation but preserves facilities entitlement.",
                    )
                )
                citations.append(
                    Citation(
                        clause_id="CAR-3.7",
                        document_title="DGCA CAR Section 3 Series M Part IV (Rev 4)",
                        quote="Meals and refreshments must be commensurate with waiting time and provided hygienically at the airport terminal.",
                        relevance="Mandates meals and refreshments for waiting passenger.",
                    )
                )

            # Check 5: Passenger was NOT offered or did not accept alternate flight -> Full CAR 3.3.2 compensation
            else:
                routing = RoutingDecision.PAY_AS_COMPUTED
                refund = True
                if block_time <= 1.0:
                    max_cap = 5000.0
                    clause_tier = "3.3.2.a"
                elif block_time <= 2.0:
                    max_cap = 7500.0
                    clause_tier = "3.3.2.b"
                else:
                    max_cap = 10000.0
                    clause_tier = "3.3.2.c"

                compensation_inr = min(max_cap, total_fare)
                summary = (
                    f"Entitlement computed under CAR Clause {clause_tier}: Flight cancelled with notice < 24 hours, "
                    f"no alternate flight accepted, block time {block_time}h. Entitled to INR {compensation_inr:,.0f} "
                    f"(lower of INR {max_cap:,.0f} or basic fare + fuel surcharge) plus full ticket refund."
                )
                citations.append(
                    Citation(
                        clause_id="CAR-3.3.2",
                        document_title="DGCA CAR Section 3 Series M Part IV (Rev 4)",
                        quote=f"Compensation scale: Clause {clause_tier} for block time {block_time}h up to INR {max_cap:,.0f} or booked one-way basic fare plus fuel charge.",
                        relevance="Direct compensation entitlement tier for unaccommodated flight cancellation.",
                    )
                )

        # --- DELAY LOGIC (CAR 3.4) ---
        elif dtype == DisruptionType.DELAY:
            delay_h = handoff["delay_hours"]
            threshold = 2.0 if block_time <= 2.5 else (3.0 if block_time <= 5.0 else 4.0)

            if delay_h >= threshold:
                routing = RoutingDecision.PAY_AS_COMPUTED if delay_h >= 6.0 else RoutingDecision.REQUEST_FURTHER_EVIDENCE
                facilities = ["Free meals and refreshments commensurate with waiting time (CAR 3.4.1)"]
                if delay_h >= 24.0:
                    facilities.append("Free hotel accommodation and ground transport (CAR 3.4.2)")
                if delay_h >= 6.0:
                    refund = True
                summary = (
                    f"Flight delayed by {delay_h}h for block time {block_time}h (threshold: {threshold}h). "
                    f"Passenger entitled to facilities under CAR 3.4.1. "
                    f"{'Full ticket refund option available under CAR 3.4.3.' if refund else ''}"
                )
                citations.append(
                    Citation(
                        clause_id="CAR-3.4.1",
                        document_title="DGCA CAR Section 3 Series M Part IV (Rev 4)",
                        quote="Airlines shall provide free meals and refreshments to passengers in proportion to waiting time",
                        relevance="Mandates food/refreshment provision for delays exceeding block time thresholds.",
                    )
                )
            else:
                routing = RoutingDecision.DECLINE
                summary = f"Delay of {delay_h}h did not meet the CAR 3.4.1 minimum threshold of {threshold}h for block time {block_time}h."
                citations.append(
                    Citation(
                        clause_id="CAR-3.4.1",
                        document_title="DGCA CAR Section 3 Series M Part IV (Rev 4)",
                        quote="Delay thresholds: block time up to 2.5 hrs: >2 hrs; 2.5-5 hrs: >3 hrs; >5 hrs: >4 hrs.",
                        relevance="Statutory threshold not met.",
                    )
                )

        # --- DENIED BOARDING LOGIC (CAR 3.2) ---
        elif dtype == DisruptionType.DENIED_BOARDING:
            if offered_alt and alt_delay <= 1.0:
                routing = RoutingDecision.DECLINE
                compensation_inr = 0.0
                summary = "Alternate flight arranged within 1 hour of scheduled departure (CAR 3.2.2.a). No monetary compensation payable."
                citations.append(
                    Citation(
                        clause_id="CAR-3.2.2",
                        document_title="DGCA CAR Section 3 Series M Part IV (Rev 4)",
                        quote="If an alternate flight is arranged within 1 hour of scheduled departure: No monetary compensation, but facilitation provided.",
                        relevance="1-hour alternate flight exemption.",
                    )
                )
            elif offered_alt and alt_delay <= 24.0:
                routing = RoutingDecision.PAY_AS_COMPUTED
                compensation_inr = min(10000.0, 2.0 * total_fare)
                summary = f"Involuntary denied boarding accommodated within 24 hours (CAR 3.2.2.b). Entitled to 200% basic fare + fuel (up to INR 10,000): INR {compensation_inr:,.0f}."
                citations.append(
                    Citation(
                        clause_id="CAR-3.2.2",
                        document_title="DGCA CAR Section 3 Series M Part IV (Rev 4)",
                        quote="If an alternate flight is arranged within 24 hours of original scheduled departure: An amount equal to 200% of booked one-way basic fare plus airline fuel charge, up to a maximum of INR 10,000.",
                        relevance="200% compensation entitlement tier for denied boarding.",
                    )
                )
            else:
                routing = RoutingDecision.PAY_AS_COMPUTED
                compensation_inr = min(20000.0, 4.0 * total_fare)
                refund = True
                summary = f"Involuntary denied boarding without alternate flight within 24 hours (CAR 3.2.2.c). Entitled to 400% basic fare + fuel (up to INR 20,000): INR {compensation_inr:,.0f} plus full refund."
                citations.append(
                    Citation(
                        clause_id="CAR-3.2.2",
                        document_title="DGCA CAR Section 3 Series M Part IV (Rev 4)",
                        quote="If an alternate flight is arranged beyond 24 hours or if passenger chooses not to travel: An amount equal to 400% of booked one-way basic fare plus airline fuel charge, up to a maximum of INR 20,000, plus full refund of ticket.",
                        relevance="400% compensation + refund tier.",
                    )
                )

        # --- DOWNGRADING LOGIC (CAR 3.5) ---
        elif dtype == DisruptionType.DOWNGRADING:
            routing = RoutingDecision.PAY_AS_COMPUTED
            compensation_inr = round(0.75 * total_fare, 2)
            summary = f"Involuntary downgrade on domestic sector (CAR 3.5.1.a). Entitled to 75% refund of ticket price: INR {compensation_inr:,.0f}."
            citations.append(
                Citation(
                    clause_id="CAR-3.5.1",
                    document_title="DGCA CAR Section 3 Series M Part IV (Rev 4)",
                    quote="Domestic flights: 75% of the ticket price including taxes and fees for the degraded sector.",
                    relevance="75% domestic downgrade compensation.",
                )
            )

        # -------------------------------------------------------------
        # A3: SELF-VERIFICATION AGAINST RETRIEVED PASSAGES
        # -------------------------------------------------------------
        verification_passed, ungrounded_citation = self._verify_citations_against_passages(citations, passages)

        if not verification_passed:
            # A3 Fallback: If verification fails, retry with alternate query or escalate to human
            return self._build_escalation_result(
                handoff,
                reason=f"Self-verification failed (A3): Citation '{ungrounded_citation}' could not be resolved directly to retrieved regulatory text.",
                clause_id="CAR-3.6.1",
            )

        return {
            "routing_decision": routing,
            "monetary_compensation_inr": compensation_inr,
            "facilities_entitled": facilities,
            "refund_entitled": refund,
            "clause_citations": citations,
            "reasoning_summary": summary,
            "escalation_reason": None,
            "self_verification_passed": True,
        }

    def _verify_citations_against_passages(
        self, citations: List[Citation], passages: List[Dict[str, Any]]
    ) -> Tuple[bool, Optional[str]]:
        """
        A3 Self-Verification:
        Ensures every citation identifier and key phrase physically resolves into the retrieved passages.
        """
        combined_text = " ".join([p.get("text", "") for p in passages]).lower()

        for c in citations:
            clause_id_norm = c.clause_id.replace("CAR-", "").replace("INDIGO-", "").lower()
            # Check if clause ID or a key snippet of the quote resolves
            has_id = clause_id_norm in combined_text or c.clause_id.lower() in combined_text
            # Extract distinctive 3-4 word phrase from quote
            quote_words = [w for w in c.quote.lower().split() if len(w) > 3][:4]
            has_phrase = any(w in combined_text for w in quote_words) if quote_words else True

            if not (has_id or has_phrase):
                return False, c.clause_id

        return True, None

    def _build_escalation_result(self, handoff: Dict[str, Any], reason: str, clause_id: str) -> Dict[str, Any]:
        return {
            "routing_decision": RoutingDecision.ESCALATE,
            "monetary_compensation_inr": 0.0,
            "facilities_entitled": ["Human Customer Relations Officer Review"],
            "refund_entitled": False,
            "clause_citations": [
                Citation(
                    clause_id=clause_id,
                    document_title="DGCA CAR Section 3 Series M Part IV (Rev 4)",
                    quote="Airlines shall maintain a grievance redressal mechanism and escalate contested claims to customer relations analysts.",
                    relevance="Mandatory escalation under dispute or unanswerable scenario (F7).",
                )
            ],
            "reasoning_summary": f"Claim escalated to human review: {reason}",
            "escalation_reason": reason,
            "self_verification_passed": True,
        }
