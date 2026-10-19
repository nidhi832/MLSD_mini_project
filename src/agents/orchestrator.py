"""
Multi-Agent Orchestrator (A1, A2, A3, A5, L4, L5, D5).
Coordinates Agent 1 (Classification) and Agent 2 (Regulatory Entitlement & Verification),
enforces step bounds, tool call bounds, wall-clock timeout, and run reconstruction tracing.
"""

import time
import uuid
from typing import Dict, Any, Optional

from src.core.contracts import (
    PassengerClaimRequest,
    EntitlementEvaluationResult,
    RoutingDecision,
)
from src.tools.claim_tools import ClaimToolBox
from src.ingestion.vector_store import RegulatoryVectorStore
from src.agents.llm_client import LLMClient
from src.agents.classifier_agent import ClaimClassifierAgent
from src.agents.regulatory_agent import RegulatoryReasoningAgent
from src.guardrails.output_guardrails import OutputGuardrails
from prompts.registry import get_prompts_by_version


class MultiAgentTriageOrchestrator:
    def __init__(
        self,
        vector_store: RegulatoryVectorStore,
        llm_client: Optional[LLMClient] = None,
        max_steps: int = 8,
        max_tool_calls: int = 10,
        wall_clock_timeout_seconds: float = 45.0,
    ):
        self.vector_store = vector_store
        self.llm_client = llm_client or LLMClient()
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls
        self.wall_clock_timeout = wall_clock_timeout_seconds

    def execute_claim_triage(
        self,
        claim: PassengerClaimRequest,
        correlation_id: Optional[str] = None,
        prompt_version: str = "v2.0",
    ) -> EntitlementEvaluationResult:
        """
        Executes bounded multi-agent pipeline with hand-off and verification.
        """
        start_time = time.time()
        run_id = f"run-{uuid.uuid4().hex[:12]}"
        corr_id = correlation_id or f"corr-{uuid.uuid4().hex[:8]}"

        prompts = get_prompts_by_version(prompt_version)
        toolbox = ClaimToolBox(claim_request=claim, vector_store=self.vector_store)

        steps_executed = 0

        try:
            # -------------------------------------------------------------
            # STEP 1: Agent 1 (Classification & Entity Extraction)
            # -------------------------------------------------------------
            steps_executed += 1
            if time.time() - start_time > self.wall_clock_timeout:
                raise TimeoutError("Wall-clock timeout exceeded before Agent 1 completion.")

            agent1 = ClaimClassifierAgent(
                llm_client=self.llm_client,
                system_prompt=prompts["classifier"],
            )
            handoff_payload = agent1.process(claim)

            # -------------------------------------------------------------
            # STEP 2: Agent 2 (Regulatory Entitlement & Self-Verification)
            # -------------------------------------------------------------
            steps_executed += 1
            if time.time() - start_time > self.wall_clock_timeout:
                raise TimeoutError("Wall-clock timeout exceeded before Agent 2 completion.")

            if toolbox.get_total_calls() >= self.max_tool_calls:
                raise RuntimeError(f"Exceeded max tool calls limit ({self.max_tool_calls})")

            agent2 = RegulatoryReasoningAgent(
                llm_client=self.llm_client,
                system_prompt=prompts["reasoner"],
                toolbox=toolbox,
            )

            decision_dict = agent2.evaluate_entitlement(handoff_payload)
            steps_executed += 1

            # -------------------------------------------------------------
            # STEP 3: L4 Output Guardrail Filtering
            # -------------------------------------------------------------
            raw_summary = decision_dict["reasoning_summary"]
            is_valid, violation, sanitized_summary = OutputGuardrails.filter_output(raw_summary)

            elapsed_total = time.time() - start_time
            cost_data = self.llm_client.cost_tracker.get_totals()

            return EntitlementEvaluationResult(
                run_id=run_id,
                correlation_id=corr_id,
                claim_id=claim.claim_id,
                pnr=claim.pnr,
                disruption_classification=handoff_payload["disruption_type"],
                extracted_entities={
                    "flight_number": handoff_payload["flight_number"],
                    "operating_carrier": handoff_payload["operating_carrier"],
                    "block_time_hours": handoff_payload["block_time_hours"],
                    "notice_hours_before_departure": handoff_payload["notice_hours_before_departure"],
                    "alternate_flight_offered": handoff_payload["offered_alternate_flight"],
                    "alternate_flight_accepted": handoff_payload["accepted_alternate_flight"],
                    "contact_details_at_booking": handoff_payload["contact_details_provided_at_booking"],
                    "cause_attribution": handoff_payload["cause_attribution"],
                },
                routing_decision=decision_dict["routing_decision"],
                monetary_compensation_inr=decision_dict["monetary_compensation_inr"],
                facilities_entitled=decision_dict["facilities_entitled"],
                refund_entitled=decision_dict["refund_entitled"],
                clause_citations=decision_dict["clause_citations"],
                reasoning_summary=sanitized_summary,
                escalation_reason=decision_dict.get("escalation_reason"),
                prompt_version=prompt_version,
                self_verification_passed=decision_dict.get("self_verification_passed", True),
                steps_executed=steps_executed,
                tool_calls_made=toolbox.get_total_calls(),
                tokens_prompt=cost_data["prompt_tokens"],
                tokens_completion=cost_data["completion_tokens"],
                cost_usd=cost_data["total_cost_usd"],
                execution_time_seconds=round(elapsed_total, 4),
            )

        except Exception as e:
            # A2/A5 Graceful Mid-Run Failure Degradation: Escalate rather than aborting or outputting unverified answer
            elapsed_total = time.time() - start_time
            cost_data = self.llm_client.cost_tracker.get_totals()

            return EntitlementEvaluationResult(
                run_id=run_id,
                correlation_id=corr_id,
                claim_id=claim.claim_id,
                pnr=claim.pnr,
                disruption_classification=claim.disruption_type,
                extracted_entities={"error": str(e)},
                routing_decision=RoutingDecision.ESCALATE,
                monetary_compensation_inr=0.0,
                facilities_entitled=["Human Case Review"],
                refund_entitled=False,
                clause_citations=[],
                reasoning_summary=f"Graceful degradation triggered mid-run due to: {type(e).__name__} ({str(e)}). Escalated to human operator.",
                escalation_reason=f"System exception mid-run: {str(e)}",
                prompt_version=prompt_version,
                self_verification_passed=False,
                steps_executed=steps_executed,
                tool_calls_made=toolbox.get_total_calls(),
                tokens_prompt=cost_data.get("prompt_tokens", 0),
                tokens_completion=cost_data.get("completion_tokens", 0),
                cost_usd=cost_data.get("total_cost_usd", 0.0),
                execution_time_seconds=round(elapsed_total, 4),
            )
