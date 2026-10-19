# System Architecture & Design Document (UE24AM342AA5 Project 04)

## 1. Executive Summary & Problem Scope
Airlines receive high volumes of passenger claims for flight cancellations, delays, denied boarding, and downgrading. Under Directorate General of Civil Aviation (DGCA) regulations (CAR Section 3, Series M, Part IV, Rev 4), entitlements are strictly governed by statutory provisions.
Ordinary RAG pipelines fail because compensation tables are accompanied by surrounding conditions and exemptions (such as alternate flight acceptance, advance notice windows, and missing passenger contact details).

This service delivers an asynchronous, resilient multi-agent triage system that evaluates passenger claims against published CAR provisions, performs rigorous self-verification, and issues grounded recommendations with clause-level citations.

---

## 2. Multi-Agent Topology & Defined Hand-Off (A1, A2, A3)

The reasoning layer consists of two distinct agents with strict responsibilities:

```
[ Incoming Passenger Claim ]
            │
            ▼
┌──────────────────────────────────────────────┐
│  Agent 1: Claim Intake & Extraction Agent    │
│  - Classifies disruption type                │
│  - Extracts block time, notice hours,        │
│    alternate flight entities, contact info   │
└──────────────────────┬───────────────────────┘
                       │ Hand-off (Structured Payload)
                       ▼
┌──────────────────────────────────────────────┐
│  Agent 2: Regulatory Entitlement &           │
│           Self-Verification Agent            │
│  - Tool 1: corpus_search                     │
│  - Tool 2: lookup_clause_by_id               │
│  - Tool 3: get_claim_field                   │
│                                              │
│  [ Self-Verification Loop (A3) ]             │
│  - Verifies every cited clause physically    │
│    resolves into retrieved passages          │
│  - Fallback / Escalation on ungrounded claim │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Output Guardrail & Safety Filter (L4)       │
│  - Suppresses unauthorized payment mandates  │
│  - Normalizes advisory language              │
└──────────────────────────────────────────────┘
```

### Loop Limits (A1)
- **Max Steps**: 8 reasoning steps.
- **Max Tool Calls**: 10 invocations.
- **Wall-Clock Timeout**: 45.0 seconds.

### Mid-Run Failure & Degradation Path (A2, A5)
If either agent fails mid-run (e.g. LLM timeout, circuit breaker open, tool error), the orchestrator catches the exception and immediately invokes the degradation path:
- Sets `routing_decision = "escalate"`
- Sets `self_verification_passed = False`
- Attaches `escalation_reason = "System exception mid-run: <details>"`
- **Critical Safety Guarantee**: An incomplete or failed run is NEVER returned as a confident answer.

---

## 3. Justification of Deployment Boundaries (D3)

The architecture partitions services across Docker containers:

1. **`api` (FastAPI Gateway)**:
   - *Rationale*: Independent scaling for public traffic ingestion, high I/O throughput, schema validation, quarantine handling, and static client serving.
2. **`agent-worker` (Stateless Reasoning Worker)**:
   - *Rationale*: CPU/Memory-intensive multi-agent reasoning, vector searching, and model interactions. Decoupled from the API gateway so heavy reasoning tasks do not degrade HTTP ingestion throughput. Scalable independently (`--scale agent-worker=3`).
3. **`redis` (Task Queue & Cache)**:
   - *Rationale*: Provides failure isolation and persistence between submission and processing. Enables asynchronous polling and token caching.
4. **`prometheus` & `grafana` (Observability)**:
   - *Rationale*: Independent lifecycle monitoring without impacting business request execution.
5. **`jaeger` (Tracing)**:
   - *Rationale*: Distributed context tracking with zero impact on operational memory.

---

## 4. Section 4.1 Difficult Case Solution

**Scenario**: Passenger flight cancelled at airport (90 min notice). Alternate flight arranged 4 hours later and accepted. Passenger demands cancellation compensation.

**Regulatory Resolution**:
1. Paragraph 3.3.2 lists compensation up to INR 10,000 based on block time.
2. However, paragraph 3.3.2 explicitly dictates:
   > *"Where the passenger accepts an alternate flight provided by the airline, the passenger shall NOT be entitled to financial compensation for cancellation under Clause 3.3.2. However, the airline remains strictly obligated to provide free of charge meals, refreshments, and facilities under Clause 3.7 and 3.8 during the waiting period."*
3. Agent 2 checks the modifier, sets `monetary_compensation_inr = 0.0`, routes to `request-further-evidence` for unredeemed refreshment expenses, and cites `CAR-3.3.2` and `CAR-3.7`.

---

## 5. Token & Cost Analysis (L5)

Based on standard lightweight model token pricing ($0.15 / 1M prompt tokens, $0.60 / 1M completion tokens):
- **Average Prompt Tokens per run**: ~1,850 tokens (Corpus chunks + claim narrative + instructions).
- **Average Completion Tokens per run**: ~120 tokens.
- **Average Cost per Run**: ~$0.00035 USD.
- **Cost per 100 Runs**: **~$0.035 USD** (~₹2.90 INR).
- **Largest Single Cost Driver**: Multi-chunk retrieval injection in Agent 2 prompt (~78% of total run token volume). Mitigated by whole-run caching (D7).

---

## 6. Service Level Objective (SLO) Justification (D6)

- **Declared Target**: **p95 Latency <= 5.0 seconds** for a complete asynchronous run (submission to final result).
- **Justification**: Section 1 and airline customer relations service level agreements require real-time claim triage response on self-service portals to prevent customer abandonment and repeat ticket submissions.
- **Load Test Strategy**: Run with Locust using LLM stub mode returning deterministic responses after realistic simulated network delays (50ms).
