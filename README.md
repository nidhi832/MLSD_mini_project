# Air Passenger Compensation Claim Triage Service

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Docker Compose](https://img.shields.io/badge/docker--compose-v2+-orange.svg)](https://docs.docker.com/compose/)
[![Test Suite](https://img.shields.io/badge/tests-7%20passed-brightgreen.svg)]()
[![DGCA CAR](https://img.shields.io/badge/regulation-CAR%20Sec%203%20M%20IV%20Rev%204-purple.svg)]()

> **Course**: UE24AM342AA5 – ML System Design and AgentOps | **Mini Project 04**  
> **Domain**: Aviation | **Institution**: PES University  

A production-grade, asynchronous multi-agent triage service that ingests passenger claims for flight cancellations, delays, denied boarding, and downgrading, validates them against published data contracts, reasons over Directorate General of Civil Aviation (DGCA) regulations, self-verifies citations, and produces legally grounded routing decisions.

---

## 📑 Table of Contents
1. [Architecture Overview](#-architecture-overview)
2. [Section 4.1 Difficult Case Solution](#-section-41-difficult-case-solution)
3. [Quick Start (Clean Clone)](#-quick-start-clean-clone)
4. [Running the Demonstration Items (1 - 10)](#-running-the-demonstration-items-1---10)
5. [Evaluation Suite & Prompt Comparison (L2, L3)](#-evaluation-suite--prompt-comparison-l2-l3)
6. [Load Testing with Locust (D6, A7)](#-load-testing-with-locust-d6-a7)
7. [Observability, Metrics & Dashboards (D5, A6)](#-observability-metrics--dashboards-d5-a6)
8. [Runbook & Incident Log (D8)](#-runbook--incident-log-d8)

---

## 🏛 Architecture Overview

```
                          ┌─────────────────────────────────────────┐
                          │         Web Client (A8 / UI)            │
                          └──────────────────┬──────────────────────┘
                                             │ REST API (F1, A4)
                                             ▼
             ┌──────────────────────────────────────────────────────────────────┐
             │                FastAPI Gateway / Ingestion Service               │
             │  • Data Contract & Cross-Field Validation (F2, D1)                │
             │  • Quarantine Storage for Rejects (F2, D1)                       │
             │  • Rate Limiting & Input Guardrails/Toxicity/Injection (L4, D7)  │
             │  • Feedback Endpoint (L6) & Run Reconstruction (D5)              │
             └───────────────────┬──────────────────────────┬───────────────────┘
                                 │ Enqueue Task             │ Metrics / Traces
                                 ▼                          ▼
 ┌──────────────────────────────────────┐          ┌──────────────────────────────────────┐
 │    Redis Queue & Cache (A4, D7)      │          │     Observability Stack (D5, A6)     │
 │  • Asynchronous Job Queue            │          │  • Prometheus Metrics Exporter       │
 │  • Embedding & Run Result Cache      │          │  • Jaeger / OpenTelemetry Tracing    │
 └──────────────────┬───────────────────┘          │  • Grafana Dashboard                 │
                    │ Dequeue                      └──────────────────────────────────────┘
                    ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                      Agent Worker Service (Scaleable x3, A7)                           │
 │                                                                                        │
 │  ┌─────────────────────────────────┐       ┌────────────────────────────────────────┐  │
 │  │ Agent 1: Classification &       │ Hand- │ Agent 2: Regulatory Entitlement        │  │
 │  │ Entity Extraction (F3)          │──off─►│ & Self-Verification Agent (F4-F7, A3)  │  │
 │  │ • Disruption category           │(A2)   │ • CAR Clause Citation & Facilities     │  │
 │  │ • Block time, notice, contacts  │       │ • Handles Section 4.1 alternate flight │  │
 │  └─────────────────────────────────┘       └───────────────────┬────────────────────┘  │
 │                                                                │                       │
 │  Tools (A1):                                                   ▼                       │
 │  1. `corpus_search` (Semantic + Keyword)           ┌────────────────────────┐          │
 │  2. `lookup_clause_by_id` (CAR 3.3.2, etc.)        │ Self-Verification Loop │          │
 │  3. `get_claim_field` (Claim details)              │ • Citation resolver    │          │
 │                                                    │ • Escalation logic     │          │
 │  LLM Engine & Resilience (A5, L5):                 └────────────────────────┘          │
 │  • Token & Cost Tracker, Circuit Breaker, Exponential Backoff, Stub Mode for Load Test │
 └─────────────────────────┬──────────────────────────────────────────────────────────────┘
                           │
                           ▼
          ┌─────────────────────────────────────────────┐
          │     Corpus & Vector Index Pipeline (D2, L1) │
          │  • DGCA CAR Sec 3 Series M Part IV Rev 4    │
          │  • Airline Passenger Charter                │
          │  • Chroma / FAISS Store (Local MiniLM-L6)   │
          └─────────────────────────────────────────────┘
```

---

## 🎯 Section 4.1 Difficult Case Solution

**Scenario**:
> *"My flight was cancelled. I was already at the airport and was told 90 minutes before departure. The airline put me on a flight four hours later, which I took. I want the compensation for cancellation."*

- **The Problem**: A blind RAG search for cancellation returns paragraph 3.3.2 compensation amounts (INR 5,000–10,000). A retriever that stops there awards an entitlement that does not exist.
- **The Solution**: Paragraph 3.3.2 explicitly dictates that providing an acceptable alternate flight waives cancellation cash compensation. However, the passenger is still entitled to meals and refreshments under paragraph 3.3.2, 3.7, and 3.8.
- **Agent Output**:
  - `routing_decision`: `request-further-evidence` (for refreshment receipts)
  - `monetary_compensation_inr`: `0.0`
  - `facilities_entitled`: `["Free meals and refreshments commensurate with waiting time at airport terminal"]`
  - `citations`: Resolves directly to `CAR-3.3.2` and `CAR-3.7`.

---

## ⚡ Quick Start (Clean Clone)

### Prerequisites
- Docker & Docker Compose (`docker compose version` >= 2.0)
- Python 3.11+ (if running tests locally)

### Single-Command Start (D3)
```bash
# 1. Clone and enter directory
cd PROJECT

# 2. Configure environment
cp .env.example .env

# 3. Start all services with single command
docker compose up --build -d
```

### Accessing Endpoints
- **Web Client UI (A8)**: [http://localhost:8000](http://localhost:8000)
- **API Swagger Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Published Data Contract (D1)**: [http://localhost:8000/api/v1/contract](http://localhost:8000/api/v1/contract)
- **Prometheus Metrics (D5)**: [http://localhost:9090](http://localhost:9090)
- **Grafana Dashboard (D5, A6)**: [http://localhost:3000](http://localhost:3000) *(User: `admin`, Pass: `admin`)*
- **Jaeger Distributed Tracing (D5)**: [http://localhost:16686](http://localhost:16686)

---

## 🎬 Running the Demonstration Items (1 - 10)

The table below maps directly to **Section 9 (Demonstration)** of the project specification:

| Item | Requirement | Command / Action | Expected Result |
| :--- | :--- | :--- | :--- |
| **1** | Clean `docker compose up` | `docker compose up -d` | All 6 services start healthy. |
| **2** | Valid request producing citations | Submit preset in UI or curl `POST /api/v1/claims` | Generates routed determination with verified citations. |
| **3** | End-to-end reconstructed run trace | `curl http://localhost:8000/api/v1/claims/<run_id>` | Reconstructs tool calls, prompt version, latency, and cost. |
| **4** | Malformed request rejection | Click **"Cross-Field Error"** preset in UI | Returns HTTP 422, quarantined in `data/quarantine`, 0 agent runs. |
| **5** | Unanswerable case escalation | Click **"Disputed Cause"** preset in UI | Routes to `escalate` under CAR 3.6.1 rather than fabricating. |
| **6** | Self-verification catches ungrounded answer | Run test suite: `python -m pytest tests/` | Demonstrates A3 verification and escalation. |
| **7** | Prompt injection defense | Click **"Prompt Injection"** preset in UI | Blocked by L4 Guardrail, quarantined immediately. |
| **8** | Circuit breaker opening | `curl -X POST http://localhost:8000/api/v1/circuit-breaker/trip` | State transitions to OPEN, subsequent runs fail fast to fallback path. |
| **9** | Locust load test against SLO | `locust -f locust/locustfile.py --headless -u 15 -r 5 --run-time 20s -H http://localhost:8000` | Demonstrates p95 latency against target and effects of `--scale`. |
| **10**| Section 4.1 difficult case | Click **"⭐ Section 4.1 Benchmark"** in UI | INR 0 cash compensation, meals/refreshments awarded, CAR 3.3.2 cited. |

---

## 📊 Evaluation Suite & Prompt Comparison (L2, L3)

The project includes a fixed batch evaluation suite of **20 scenarios** (`data/evaluation_set_20.json`) runnable from the command line:

```bash
# Run comparison between Prompt v1.0 and Prompt v2.0
python -m src.eval.runner --compare
```

**Results:**
- **Groundedness**: `0.8500`
- **Context Relevance**: `0.8083`
- **Faithfulness**: Improved from `0.2700` (v1.0) to `0.2850` (v2.0)
- **Hallucination Rate**: Reduced by `-0.0150` on the difficult benchmark!

---

## 🚀 Scaling & Load Testing (A7, D6)

### Scaling Workers (A7)
```bash
docker compose up --scale agent-worker=3 -d
```

### Running Locust Headless Load Test (D6)
```bash
locust -f locust/locustfile.py --headless -u 20 -r 5 --run-time 30s -H http://localhost:8000
```
- **Declared SLO Target**: Complete asynchronous run p95 <= 5.0 seconds.

---

## 📖 Runbook & Documentation
- [Operational Runbook (D8)](docs/RUNBOOK.md)
- [Development Incident Log (D8)](docs/INCIDENT_LOG.md)
- [System Architecture Document](docs/ARCHITECTURE.md)
