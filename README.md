# Air Passenger Compensation Claim Triage Service

> **Course**: UE24AM342AA5 – ML System Design and AgentOps | **Mini Project 04** | PES University
> **Domain**: Aviation — DGCA CAR Section 3, Series M, Part IV (Rev 4)

A multi-agent service that accepts a passenger claim (cancellation, delay, denied boarding,
downgrading), validates it, works out the entitlement from the CAR and the airline charter,
verifies every citation against the text it actually retrieved, and routes the claim to
**pay-as-computed**, **request-further-evidence**, **decline** or **escalate** with a cited basis.
It never authorises a payment; a customer relations officer decides.

---

## Contents
1. [Quick start (clean clone)](#1-quick-start-clean-clone)
2. [Services](#2-services)
3. [How the agents work](#3-how-the-agents-work)
4. [Using a real LLM (optional)](#4-using-a-real-llm-optional)
5. [Demonstration items 1–10](#5-demonstration-items-110)
6. [Evaluation (L2, L3)](#6-evaluation-l2-l3)
7. [Load testing and scaling (D6, A7)](#7-load-testing-and-scaling-d6-a7)
8. [Observability (D5, A6)](#8-observability-d5-a6)
9. [Configuration](#9-configuration)
10. [Repository layout and docs](#10-repository-layout-and-docs)

---

## 1. Quick start (clean clone)

**Prerequisites:** Git and Docker Desktop (Compose v2). Docker Desktop must be open and show *Engine running*.
Allow about 8 GB of free disk space. Nothing else needs to be installed: everything runs in containers.

```bash
git clone https://github.com/nidhi832/MLSD_mini_project.git
cd MLSD_mini_project

# Create your local config (Windows Command Prompt: copy .env.example .env)
cp .env.example .env

# Build once and start all services
docker compose up --build -d

# Check: 8 services Up; api, redis and the agents show (healthy)
docker compose ps
```

The first build takes 10–20 minutes (Python libraries, CPU-only PyTorch, the embedding model).
Later starts take seconds: `docker compose up -d`. Stop with `docker compose down`.

Open **http://localhost:8000**, click **Find out how much you're owed**, choose the
**⭐ Section 4.1 Benchmark** preset and click **Run Multi-Agent Triage**.
Expected: *Request Further Evidence*, ₹0, meals and refreshments, citations CAR-3.3.2 and CAR-3.7, verification PASSED.

| What | URL |
|---|---|
| Web client (A8) | http://localhost:8000 |
| API docs (Swagger) | http://localhost:8000/docs |
| Data contract (D1) | http://localhost:8000/api/v1/contract |
| Health (F8) | http://localhost:8000/health |
| Grafana (user `admin`, password from `.env`, default `admin`) | http://localhost:3000 |
| Prometheus | http://localhost:9090 |
| Jaeger traces | http://localhost:16686 |

**Troubleshooting**
- `failed to connect to the docker API` → Docker Desktop is not running; open it and wait for *Engine running*.
- `port is already allocated` → another container uses 8000/3000/6379/9090; `docker ps` and stop it.
- Build stops with `EOF` / connection reset → network drop; run `docker compose up --build -d` again (finished layers are cached).
- Docker Desktop crashes during the build → see `docs/RUNBOOK.md`, failure mode 4.

## 2. Services

| Service | Role | Why it is a separate deployment unit (D3) |
|---|---|---|
| `api` | FastAPI gateway: data contract, validation, quarantine, guardrails, rate limiting, run status, web client | Handles HTTP traffic; must stay responsive while agents run for seconds |
| `classifier-agent` | **Agent 1**: classification and entity extraction, narrative/field consistency check | Different prompt and resource profile (no embedding model); independent release |
| `agent-worker` | **Agent 2**: LangGraph regulatory agent with tools and self-verification | The bottleneck component; scaled with `--scale agent-worker=3` (A7) |
| `llm-stub` | OpenAI-compatible LLM stub with realistic delay | Lets load tests measure this system, not provider rate limits (D6); stopping it demonstrates the circuit breaker |
| `redis` | Queues, shared run records, whole-run cache, shared circuit-breaker state | Failure isolation between submission and execution (A4) |
| `prometheus`, `grafana` | Metrics and dashboard | Independent lifecycle from the business path |
| `jaeger` | OpenTelemetry trace backend | Same |
| `cadvisor` (optional) | Per-container CPU/memory | `docker compose --profile resources up -d` |

All application services use **one image** built once by `api`.

## 3. How the agents work

```
Browser / client
   │ POST /api/v1/claims  (validated against the data contract; rejects quarantined, no agent run)
   ▼
api ──► Redis claim_processing_queue ──► classifier-agent (Agent 1)
                                            │ structured hand-off (+ fact conflicts)
                                            ▼
                     Redis claim_reasoning_queue ──► agent-worker (Agent 2, LangGraph)
                                                        │ run record (full reconstruction)
                                                        ▼
                                            Redis run:<run_id>  ◄── GET /api/v1/claims/<run_id> (poll)
```

**Agent 2 graph (LangGraph):** `plan → run_tool → plan → … → decide → verify → finalize | retry | escalate`

1. **Retrieve** the provision that appears to apply (`corpus_search`).
2. **Find modifiers**: every retrieved passage is scanned for references to other provisions
   ("under Clause 3.7 and 3.8", "Provisions of 3.3.2 shall apply"); unread ones are read with `lookup_clause_by_id`.
3. **Search again** with a query derived from what it read, for conditions and exemptions.
4. **Verify**: every citation must quote a line of retrieved text. Amounts and routing are computed
   deterministically, and only provisions that were actually retrieved can be applied.
5. **Retry or escalate**: if verification fails, re-search with a different strategy (direct clause
   lookup + hybrid keyword/dense search); if it still fails, escalate. An unverified answer is never returned.

Tools (A1): `corpus_search` (FAISS via LangChain, dense / keyword / hybrid), `lookup_clause_by_id`, `get_claim_field`.
Loop bounds: `MAX_AGENT_STEPS`, `MAX_AGENT_TOOL_CALLS`, `AGENT_WALL_CLOCK_TIMEOUT_SECONDS`; hitting one escalates.
With a real LLM the model chooses each tool call; guards stop it finishing while cross-references are unread
or before the exemptions search. Without a model (stub, or provider down) a deterministic planner makes the choices.

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## 4. Using a real LLM (optional)

The default (`LLM_PROVIDER=stub-service`) needs no key. To use a free provider, edit **your `.env` only** (never `.env.example`):

```ini
# Groq (free key: https://console.groq.com/keys)
LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-20b
LLM_API_KEY=your_key_here
```
or `LLM_PROVIDER=gemini` with a current Gemini *flash* model id and a key from https://aistudio.google.com/apikey.
Apply with `docker compose up -d`. Each run's record shows `llm_provider`, `llm_model`, and in `agent_trace`
whether each step was chosen by the `llm` or the `policy`.

## 5. Demonstration items 1–10

Run commands from the project folder. On Windows Command Prompt, `curl` works as shown.

| # | Item | How |
|---|---|---|
| 1 | Clean start | `docker compose up --build -d` then `docker compose ps` (8 services) |
| 2 | Valid request with citations | Web client → **Cancellation (Eligible Pay)** preset → Run. Citation quotes are lines of retrieved text |
| 3 | Run traced end to end | `curl http://localhost:8000/api/v1/claims/<run_id>` → `agent_trace` (every step, tool, query, latency), `tool_call_log`, `retrieved_passages`, `prompt_version`, `cost_breakdown.per_step`, `verification`. In Jaeger, search service `agent-worker` (one trace spans API → Agent 1 → Agent 2 → tools) |
| 4 | Malformed request quarantined, no agent run | **Cross-Field Error (F2)** preset → HTTP 422 with `quarantine_id`; `curl http://localhost:8000/api/v1/quarantine`; `curl http://localhost:8000/api/v1/runs` shows no run was created |
| 5 | Unanswerable → escalation | **Disputed Cause (F7)** preset → *Escalate*, citing CAR 3.6.2 (burden of proof) |
| 6 | Verification catches unsupported answer | Submit any claim to `http://localhost:8000/api/v1/claims?demo_fault=unsupported_citation` (Swagger: `/docs` → POST /claims → fill `demo_fault`). The trace shows verification *failed* → re-search → *failed* → *escalate*. Also `docker compose exec api python -m pytest tests/ -k verification -v` |
| 7 | Prompt injection | **Prompt Injection (L4)** preset → HTTP 400, quarantined, no run |
| 8 | Circuit breaker + degradation | `docker compose stop llm-stub`, submit a claim, `curl http://localhost:8000/api/v1/circuit-breaker` → `OPEN`; the run completes via the fallback planner, marked `degraded_mode: true`, still verified. `docker compose start llm-stub` → closes after 30 s. (Manual: `curl -X POST .../circuit-breaker/trip` and `/reset`) |
| 9 | Locust vs SLO, failing concurrency, `--scale` | See section 7 |
| 10 | Section 4.1 case | **⭐ Section 4.1 Benchmark** preset → ₹0, meals and refreshments, CAR-3.3.2 + CAR-3.7. Expand *How the agent reached this*: it followed 3.3.2's reference to 3.7 |

Tests: `docker compose exec api python -m pytest tests/ -q` (36 tests).

## 6. Evaluation (L2, L3)

```bash
docker compose exec api python -m src.eval.runner --compare --out data/runs/eval_report.json
```
20 scenarios in `data/evaluation_set_20.json`, each with expected routing, amount and supporting clause,
including the Section 4.1 case. Two variants run on the same set:

- **v1.0 (before)**: one retrieval pass, answer from what came back, no reference following, no re-search.
- **v2.0 (after)**: the full agent loop. The deliberate change is enabling spec section 5 steps 2–5.

Metrics (definitions in `src/eval/metrics.py`): groundedness (cited quotes found verbatim in retrieved text),
context relevance (retrieved passages from the governing section), faithfulness (0.7 × routing correct +
0.3 × amount correct), hallucination rate (confident answers that are wrong or cite unretrieved text),
plus supporting-passage recall.

Result measured during development (built-in embedder; re-run in Docker for your report):

| Metric | v1.0 | v2.0 |
|---|---|---|
| Correct outcomes | 11 / 20 | 20 / 20 |
| Faithfulness | 0.61 | 1.00 |
| Hallucination rate | 0.25 | 0.00 |
| Supporting-passage recall | 0.70 | 0.85 |
| Avg tool calls per run | 1.0 | 6.4 |

Note: Scenario 20's expected outcome was corrected (72 h notice → CAR 3.3.1(b): refund or alternate flight, no compensation).

## 7. Load testing and scaling (D6, A7)

**Declared SLO (before measuring):** p95 of a complete run — submission to final result — ≤ **5.0 s**.
Justification: the triage result is shown to a passenger or analyst waiting on a portal page; beyond a
few seconds users resubmit, which multiplies agent runs and cost.

The LLM is the `llm-stub` service (`STUB_DELAY_SECONDS=0.6` per call, ~8 calls per run).

```bash
docker compose exec api locust -f locust/locustfile.py --headless -u 10 -r 2 --run-time 60s -H http://localhost:8000 --csv data/runs/locust_u10
```
Read the **E2E_RUN Claim_Submission_To_Completion** row (RPS, 50%, 95%). Repeat with `-u 5, 10, 20, 40`
to find where p95 passes 5 s, then:
```bash
docker compose up -d --scale agent-worker=3
```
and repeat. Every simulated user is a separate client with unique PNRs, so neither the rate limiter nor the
cache distorts the result. Cache effect (D7): set `CACHE_ENABLED=false` in `.env`, `docker compose up -d`,
re-run, and compare p95 and `llm_cost_usd_total`; or run with `LOCUST_ALLOW_CACHE=1` for a high hit ratio.

## 8. Observability (D5, A6)

- **Logs**: structured JSON with service name, `run_id` and `correlation_id` (`docker compose logs agent-worker`).
- **Traces**: OpenTelemetry → Jaeger; one trace per run across api, classifier-agent and agent-worker, with a span per graph node and tool call.
- **Metrics**: Prometheus scrapes the API and every agent replica (DNS discovery).
- **Grafana dashboard** *Air Passenger Claim Triage - AgentOps*: throughput, p50/p95 latency vs SLO, error
  and escalation rate, success rate per agent, steps and tool calls per run, p95 per graph step, retry rate,
  verification outcomes, availability (`up`), container uptime, CPU and memory per service, cache hit ratio,
  cost, live quality of sampled runs (L6), analyst feedback, index freshness (D2).
- **Run reconstruction**: `GET /api/v1/claims/<run_id>`; recent runs: `GET /api/v1/runs`.

## 9. Configuration

All environment-specific settings come from `.env` at runtime (template: `.env.example`, documented inline).
No credentials, endpoints or model identifiers are hard-coded (D4). Key settings: `LLM_PROVIDER`,
`LLM_MODEL`, `LLM_API_KEY`, loop bounds, `CACHE_ENABLED`, `CLIENT_RATE_LIMIT_RPS`, `STUB_DELAY_SECONDS`.

Rebuild the vector index (idempotent; only changed chunks are re-embedded):
`docker compose exec api python -m src.ingestion.pipeline` (`--force` to rebuild everything).
Freshness: `curl http://localhost:8000/api/v1/index/freshness`.

## 10. Repository layout and docs

```
src/api/            FastAPI gateway (main.py, routes.py) and agent workers (worker.py)
src/agents/         classifier_agent.py (Agent 1), regulatory_agent.py (Agent 2, LangGraph),
                    entitlement_rules.py (evidence-gated calculator), orchestrator.py, llm_client.py
src/tools/          the three agent tools
src/ingestion/      chunker, FAISS vector store, ingestion pipeline CLI
src/core/           contracts, queue, Redis shared state, cache, rate limiter, resilience, tracing, observability
src/guardrails/     input (injection, toxicity) and output (authority, toxicity) guardrails
src/eval/           metrics and batch runner
src/stub/           LLM stub service
prompts/            versioned prompts v1.0 and v2.0
data/               corpus, vector index, evaluation set, quarantine, feedback
tests/              unit and agent tests
docs/               ARCHITECTURE.md, RUNBOOK.md, INCIDENT_LOG.md
```
