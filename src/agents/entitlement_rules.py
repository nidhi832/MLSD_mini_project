"""
Evidence-gated entitlement calculator used by the Regulatory agent (F4, F5, F6, A3).

Design decision: the language model never computes money. Amounts, tiers and
routing are computed here, deterministically, from the claim facts. But this
calculator can only use what the agent actually retrieved:

* Every modifying provision (exemptions, the alternate-flight condition, the
  facilities clause, notice rules) is applied ONLY if its text is present in the
  passages the agent retrieved. If the agent never found CAR 3.3.4, the weather
  exemption is not applied — exactly the failure a naive retriever makes.
* Every citation names a clause and an anchor phrase; its quote is the line of
  retrieved text that contains the anchor. If no retrieved passage contains it,
  the citation is left unresolved and the verification step rejects the answer.

So the quality of the answer depends on the agent's retrieval decisions, and an
answer can never cite text the agent did not read.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from src.core.contracts import DisruptionType, RoutingDecision

CAR_TITLE = "DGCA CAR Section 3 Series M Part IV (Rev 4)"
CHARTER_TITLE = "IndiGo Passenger Charter 6E-COC-2023-V2"

EXTRAORDINARY_CAUSES = {"weather", "atc", "air_traffic_control", "runway_closure", "bird_strike",
                        "fog", "political_turmoil", "government_order", "security"}
CARRIERS_IN_SCOPE = {"indigo", "air india", "vistara", "spicejet"}

# Anchor phrases: exact phrases from the corpus that each rule depends on.
A = {
    "contact_exemption": ("CAR-3.3.4", "failed to provide correct contact details"),
    "extraordinary_exemption": ("CAR-3.3.4", "caused by extraordinary circumstances beyond airline control"),
    "refused_alt_exemption": ("CAR-3.3.4", "fails to accept an alternate flight departing within 2 hours"),
    "notice_2_weeks": ("CAR-3.3.1", "at least 2 weeks before scheduled departure"),
    "notice_24h_2w": ("CAR-3.3.1", "less than 2 weeks and up to 24 hours before departure"),
    "alt_accepted_condition": ("CAR-3.3.2", "where the passenger accepts an alternate flight provided by the airline"),
    "cancel_tier_a": ("CAR-3.3.2", "clause 3.3.2.a: inr 5,000"),
    "cancel_tier_b": ("CAR-3.3.2", "clause 3.3.2.b: inr 7,500"),
    "cancel_tier_c": ("CAR-3.3.2", "clause 3.3.2.c: inr 10,000"),
    "meals": ("CAR-3.7", "meals and refreshments must be commensurate with waiting time"),
    "delay_meals": ("CAR-3.4.1", "airlines shall provide free meals and refreshments to passengers in proportion to waiting time"),
    "delay_tier_a": ("CAR-3.4.1", "for flights with block time up to 2.5 hours: delay exceeding 2 hours"),
    "delay_tier_b": ("CAR-3.4.1", "for flights with block time between 2.5 hours and 5 hours: delay exceeding 3 hours"),
    "delay_tier_c": ("CAR-3.4.1", "for flights with block time exceeding 5 hours: delay exceeding 4 hours"),
    "hotel_24h": ("CAR-3.4.2", "flight departure delay is more than 24 hours beyond scheduled time"),
    "refund_6h": ("CAR-3.4.3", "if a flight delay is expected to exceed 6 hours"),
    "db_tier_a": ("CAR-3.2.2", "alternate flight is arranged within 1 hour of scheduled departure"),
    "db_tier_b": ("CAR-3.2.2", "alternate flight is arranged within 24 hours of original scheduled departure"),
    "db_tier_c": ("CAR-3.2.2", "alternate flight is arranged beyond 24 hours"),
    "downgrade_domestic": ("CAR-3.5.1", "domestic flights: 75% of the ticket price"),
    "burden_of_proof": ("CAR-3.6.2", "burden of proving"),
    "grievance": ("CAR-3.6.1", "escalate through the ministry of civil aviation airsewa portal"),
    "charter_vouchers": ("INDIGO-3.2", "monetary compensation is waived, but indigo provides airport food and beverage vouchers"),
    "charter_receipts": ("INDIGO-5.2", "expense receipts for refreshments/meals incurred during disruption"),
}


def _norm(text: str) -> str:
    text = text.lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip()


def clause_compatible(cited: str, retrieved: str) -> bool:
    """A citation of CAR-3.7 is supported by a passage CAR-3.7 or CAR-3.7.1 (and vice versa)."""
    c, r = cited.upper(), retrieved.upper()
    return c == r or r.startswith(c + ".") or c.startswith(r + ".")


class Evidence:
    """The set of passages the agent has retrieved during this run."""

    def __init__(self, passages: List[Dict[str, Any]]):
        self.passages = passages

    def find(self, clause_id: str, anchor: str) -> Optional[Tuple[Dict[str, Any], str]]:
        """Return (passage, quoted line) if a compatible retrieved passage contains the anchor."""
        a = _norm(anchor)
        for p in self.passages:
            if not clause_compatible(clause_id, p["clause_id"]):
                continue
            for line in p["text"].splitlines():
                if a in _norm(line):
                    return p, line.strip()
            if a in _norm(p["text"]):
                return p, anchor
        return None

    def has(self, key: str) -> bool:
        clause_id, anchor = A[key]
        return self.find(clause_id, anchor) is not None


def _cite(key: str, relevance: str) -> Dict[str, Any]:
    clause_id, anchor = A[key]
    return {"clause_id": clause_id, "anchor": anchor, "relevance": relevance,
            "document_title": CHARTER_TITLE if clause_id.startswith("INDIGO") else CAR_TITLE}


def decide(handoff: Dict[str, Any], evidence: Evidence, apply_modifiers: bool = True) -> Dict[str, Any]:
    """
    Compute the entitlement position. Returns a dict with routing, amounts, facilities,
    refund, a reasoning summary and citation specs (clause + anchor, resolved later).
    `apply_modifiers=False` reproduces the naive baseline used for the L3 comparison.
    """
    dtype = handoff["disruption_type"]
    if isinstance(dtype, str):
        dtype = DisruptionType(dtype)
    cause = str(handoff.get("cause_attribution", "")).lower()
    carrier = str(handoff.get("operating_carrier", "IndiGo")).lower()
    block = float(handoff["block_time_hours"])
    notice = float(handoff["notice_hours_before_departure"])
    offered = bool(handoff["offered_alternate_flight"])
    accepted = bool(handoff["accepted_alternate_flight"])
    alt_delay = float(handoff.get("alternate_flight_delay_hours") or 0.0)
    contact_ok = bool(handoff["contact_details_provided_at_booking"])
    fare = float(handoff["ticket_basic_fare_inr"]) + float(handoff.get("airline_fuel_charge_inr") or 0.0)
    gate = (lambda k: apply_modifiers and evidence.has(k))

    out = {"routing_decision": RoutingDecision.DECLINE, "monetary_compensation_inr": 0.0,
           "facilities_entitled": [], "refund_entitled": False, "citations": [],
           "reasoning_summary": "", "escalation_reason": None}

    def escalate(reason: str, keys: List[str]):
        out.update(routing_decision=RoutingDecision.ESCALATE,
                   facilities_entitled=["Human Customer Relations Officer Review"],
                   reasoning_summary=f"Escalated to a customer relations officer: {reason}",
                   escalation_reason=reason)
        out["citations"] = [_cite(k, "Basis for escalating instead of answering (F7).") for k in keys if evidence.has(k)]
        return out

    # ---------------- F7: cases the system must not answer ----------------
    conflicts = handoff.get("fact_conflicts") or []
    if conflicts:
        return escalate("The narrative contradicts the structured claim fields: " + "; ".join(conflicts), ["grievance"])
    if "disputed" in cause or "unknown" in cause or "contested" in cause:
        return escalate("Cause attribution is contested between passenger and airline; under CAR 3.6.2 the "
                        "burden of proof rests with the airline, so a human must weigh the evidence.",
                        ["burden_of_proof", "grievance"])
    if carrier not in CARRIERS_IN_SCOPE:
        return escalate(f"Operating carrier '{handoff.get('operating_carrier')}' is outside the corpus "
                        "(only the IndiGo charter and Indian scheduled carriers are covered).", ["grievance"])

    cites = out["citations"]

    # ---------------- Cancellation (CAR 3.3) ----------------
    if dtype == DisruptionType.CANCELLATION:
        if not contact_ok and gate("contact_exemption"):
            cites.append(_cite("contact_exemption", "Exemption: no compensation when contact details were not provided at booking."))
            out["reasoning_summary"] = ("Declined under CAR 3.3.4(b): the passenger did not provide correct contact details at "
                                        "booking, which removes the airline's liability for cancellation compensation.")
        elif notice >= 336.0 and gate("notice_2_weeks"):
            cites.append(_cite("notice_2_weeks", "Cancellation notified at least 2 weeks ahead: refund or alternate flight, no compensation."))
            out.update(refund_entitled=not accepted)
            out["reasoning_summary"] = ("Declined under CAR 3.3.1(a): the cancellation was notified at least two weeks before "
                                        "departure, so only an alternate flight or full refund is owed, not compensation.")
        elif cause in EXTRAORDINARY_CAUSES and gate("extraordinary_exemption"):
            cites.append(_cite("extraordinary_exemption", "Exemption for extraordinary circumstances beyond airline control."))
            out["reasoning_summary"] = (f"Declined under CAR 3.3.4(a): the cancellation was caused by extraordinary "
                                        f"circumstances beyond the airline's control ({cause}).")
        elif 24.0 <= notice < 336.0 and gate("notice_24h_2w"):
            cites.append(_cite("notice_24h_2w", "Notice between 24 hours and 2 weeks: alternate flight or full refund only."))
            out.update(refund_entitled=not accepted)
            out["reasoning_summary"] = (f"Declined under CAR 3.3.1(b): notice of {notice:g} hours (between 24 hours and 2 weeks) "
                                        "entitles the passenger to an alternate flight or a full refund, but not to compensation.")
        elif accepted and gate("alt_accepted_condition"):
            cites.append(_cite("alt_accepted_condition", "Condition in 3.3.2: accepting the alternate flight removes cash compensation."))
            if alt_delay >= 2.0 and gate("meals"):
                cites.append(_cite("meals", "Facilities provision referred to by 3.3.2: meals and refreshments while waiting."))
                facilities = ["Free meals and refreshments commensurate with waiting time at airport terminal"]
                if alt_delay >= 24.0 and gate("hotel_24h"):
                    cites.append(_cite("hotel_24h", "Hotel accommodation for waits over 24 hours."))
                    facilities.append("Free hotel accommodation and airport transfers")
                if gate("charter_receipts"):
                    cites.append(_cite("charter_receipts", "Evidence needed to reimburse meals if vouchers were not redeemed."))
                out.update(routing_decision=RoutingDecision.REQUEST_FURTHER_EVIDENCE, facilities_entitled=facilities)
                out["reasoning_summary"] = (
                    f"No cancellation compensation is payable: under CAR 3.3.2 the airline may provide either an acceptable "
                    f"alternate flight or compensation, and the passenger accepted the alternate flight ({alt_delay:g} hours later). "
                    f"The passenger is still owed meals and refreshments for the wait under the facilities provision (CAR 3.7) "
                    f"that 3.3.2 refers to. Routed to request further evidence: meal receipts if vouchers were not provided.")
            else:
                out["reasoning_summary"] = ("Declined under CAR 3.3.2: the passenger accepted the alternate flight, so cancellation "
                                            "compensation is not payable.")
        elif offered and not accepted and alt_delay <= 2.0 and gate("refused_alt_exemption"):
            cites.append(_cite("refused_alt_exemption", "Exemption: passenger refused an alternate flight within 2 hours."))
            out.update(refund_entitled=True)
            out["reasoning_summary"] = ("Declined under CAR 3.3.4(c): the passenger did not accept an alternate flight departing "
                                        "within 2 hours of the original departure.")
        else:
            tier_key, cap, label = (("cancel_tier_a", 5000.0, "3.3.2.a") if block <= 1.0 else
                                    ("cancel_tier_b", 7500.0, "3.3.2.b") if block <= 2.0 else
                                    ("cancel_tier_c", 10000.0, "3.3.2.c"))
            comp = min(cap, fare)
            cites.append(_cite(tier_key, f"Compensation tier for block time {block:g} h."))
            out.update(routing_decision=RoutingDecision.PAY_AS_COMPUTED, monetary_compensation_inr=comp, refund_entitled=True)
            out["reasoning_summary"] = (f"Entitlement computed under CAR clause {label}: cancellation notified less than 24 hours "
                                        f"before departure, no alternate flight accepted, block time {block:g} h. Compensation is "
                                        f"INR {comp:,.0f} (the lower of INR {cap:,.0f} and basic fare plus fuel charge) plus a full refund.")

    # ---------------- Delay (CAR 3.4) ----------------
    elif dtype == DisruptionType.DELAY:
        delay = float(handoff.get("delay_hours") or 0.0)
        tier_key, threshold = (("delay_tier_a", 2.0) if block <= 2.5 else
                               ("delay_tier_b", 3.0) if block <= 5.0 else ("delay_tier_c", 4.0))
        cites.append(_cite(tier_key, f"Delay threshold for block time {block:g} h is more than {threshold:g} h."))
        if delay > threshold:
            cites.append(_cite("delay_meals", "Meals and refreshments in proportion to waiting time."))
            facilities = ["Free meals and refreshments commensurate with waiting time (CAR 3.4.1)"]
            if delay > 24.0:
                cites.append(_cite("hotel_24h", "Hotel and transport for delays over 24 hours."))
                facilities.append("Free hotel accommodation and ground transport (CAR 3.4.2)")
            refund = delay > 6.0
            if refund:
                cites.append(_cite("refund_6h", "Delays over 6 hours: alternate flight or full refund."))
            out.update(routing_decision=RoutingDecision.PAY_AS_COMPUTED if refund else RoutingDecision.REQUEST_FURTHER_EVIDENCE,
                       facilities_entitled=facilities, refund_entitled=refund)
            out["reasoning_summary"] = (f"Flight delayed {delay:g} h on a {block:g} h block-time flight, above the {threshold:g} h "
                                        f"threshold of CAR 3.4.1, so meals and refreshments are owed."
                                        + (" Delay over 6 hours: alternate flight or full refund under CAR 3.4.3." if refund else ""))
        else:
            out["reasoning_summary"] = (f"Declined: a delay of {delay:g} h does not exceed the CAR 3.4.1 threshold of "
                                        f"{threshold:g} h for a {block:g} h block-time flight.")

    # ---------------- Denied boarding (CAR 3.2) ----------------
    elif dtype == DisruptionType.DENIED_BOARDING:
        if offered and alt_delay <= 1.0:
            cites.append(_cite("db_tier_a", "Alternate flight within 1 hour: no monetary compensation."))
            out["reasoning_summary"] = "Declined under CAR 3.2.2(a): an alternate flight was arranged within 1 hour of departure."
        elif offered and alt_delay <= 24.0:
            comp = min(10000.0, 2.0 * fare)
            cites.append(_cite("db_tier_b", "200% of basic fare plus fuel charge, capped at INR 10,000."))
            out.update(routing_decision=RoutingDecision.PAY_AS_COMPUTED, monetary_compensation_inr=comp)
            out["reasoning_summary"] = (f"Involuntary denied boarding with an alternate flight within 24 hours (CAR 3.2.2(b)): "
                                        f"200% of basic fare plus fuel, capped at INR 10,000 = INR {comp:,.0f}.")
        else:
            comp = min(20000.0, 4.0 * fare)
            cites.append(_cite("db_tier_c", "400% of basic fare plus fuel charge, capped at INR 20,000, plus refund."))
            out.update(routing_decision=RoutingDecision.PAY_AS_COMPUTED, monetary_compensation_inr=comp, refund_entitled=True)
            out["reasoning_summary"] = (f"Involuntary denied boarding without an alternate flight within 24 hours (CAR 3.2.2(c)): "
                                        f"400% of basic fare plus fuel, capped at INR 20,000 = INR {comp:,.0f}, plus a full refund.")

    # ---------------- Downgrading (CAR 3.5) ----------------
    elif dtype == DisruptionType.DOWNGRADING:
        comp = round(0.75 * fare, 2)
        cites.append(_cite("downgrade_domestic", "Domestic downgrade: 75% of the ticket price."))
        out.update(routing_decision=RoutingDecision.PAY_AS_COMPUTED, monetary_compensation_inr=comp)
        out["reasoning_summary"] = f"Involuntary downgrade on a domestic sector (CAR 3.5.1(a)): 75% of the ticket price = INR {comp:,.0f}."

    return out
