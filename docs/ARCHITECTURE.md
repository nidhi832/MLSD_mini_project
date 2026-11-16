# System Architecture & Design (UE24AM342AA5 Mini Project 04)

## 1. Problem and scope
Airline passenger claims (cancellation, delay, denied boarding, downgrading) must be assessed against
DGCA CAR Section 3, Series M, Part IV (Rev 4) and the airline charter (IndiGo). The amounts are easy to
retrieve; the conditions that decide whether they are payable sit around them and refer to other clauses.
A retriever that finds the amounts and stops states entitlements that do not exist (spec section 3, 4.1).

The service produces an entitlement **recommendation** with citations and routes it; it never authorises payment.

## 2. Deployment topology and boundaries (D3)

```
                      ┌──────────────────── docker compose ─────────────────────────────┐
 browser/client ──►   │ api (FastAPI) ──► redis ◄──► classifier-agent ──► redis ◄──► agent-worker ×N │
                      │     ▲  run records / cache / circuit state in redis      │            │
                      │     └──────────── poll GET /claims/<run_id> ◄────────────┘            │
                      │ agent-worker, classifier-agent ──HTTP──► llm-stub | Groq | Gemini      │
                      │ all app services ──OTLP──► jaeger      prometheus ──scrape──► all      │
                      │ grafana ──► prometheus                 (optional) cadvisor             │
                      └─────────────────────────────────────────────────────────────────────┘
```

| Boundary | Justification |
|---|---|
| api ↔ agents | Runs take seconds; the API returns a run id immediately (A4) and stays responsive. Failure of an agent never takes down ingestion. |
| Agent 1 ↔ Agent 2 | Different responsibilities, prompts and resource profile (Agent 1 needs no embedding model). Agent 2 is the bottleneck and scales alone (A7). Independent release. |
| redis | The queue isolates submission from execution; shared state lets any replica serve any run, read the cache, and see the same circuit breaker. |
| llm-stub | Replaces the provider for load tests (D6) and is the component stopped to demonstrate the circuit breaker (A5). |
| observability services | Own lifecycle; no effect on the request path. |

**Asynchronous choice (A4):** submit returns `202` with `run_id`; the client polls `GET /api/v1/claims/<run_id>`.
Polling was chosen over callbacks because the web client and Locust both run behind no public endpoint,
and polling needs no client-side server. A Redis list is the queue between submission and execution.

## 3. Agents (A1, A2)

**Agent 1 – classifier-agent** (`src/agents/classifier_agent.py`): extracts disruption category, notice,
block time, alternate-flight and contact entities from the narrative (LLM, or keywords in stub mode) and
records contradictions between the narrative and the form fields (`fact_conflicts`). Hands off a structured
payload via `claim_reasoning_queue`.

**Agent 2 – agent-worker** (`src/agents/regulatory_agent.py`, LangGraph):

```
START → plan ─(tool)→ run_tool → plan … ─(finish)→ decide → verify ─pass→ finalize → END
                                                               ├─retry→ plan (different strategy)
                                                               └─fail→ escalate → END
```
- `plan`: with a real LLM the model picks the next tool and arguments; guards prevent finishing while
  cross-references are unread or before the modifier search. Without a model, a deterministic policy planner.
- `run_tool`: executes the tool, merges evidence, extracts references to other clauses from new passages.
- `decide`: `entitlement_rules.py` computes routing and amounts; every modifying provision is applied only
  if its text is in the retrieved evidence, and each citation quotes a retrieved line verbatim.
- `verify` (A3): all citations must resolve to retrieved text; a payable amount must be supported; a run
  stopped by a loop bound fails. Up to 2 re-searches (direct clause lookup + hybrid search), then escalation.
- LLM explanations are accepted only if every clause number they mention was retrieved.

**Tools:** `corpus_search(query, strategy=dense|keyword|hybrid)`, `lookup_clause_by_id(clause_id)` (returns the
clause with its sub-clauses), `get_claim_field(field_name)`.

**Loop bounds:** `MAX_AGENT_STEPS=16`, `MAX_AGENT_TOOL_CALLS=10`, `AGENT_WALL_CLOCK_TIMEOUT_SECONDS=45`.

**Failure mid-run (A2, A5):**
| Failure | Behaviour |
|---|---|
| Agent 1 exception | Run completed as ESCALATE with the error recorded |
| Agent 2 exception | Run completed as ESCALATE with the error recorded |
| Provider down / circuit OPEN | Agent 1 falls back to keyword extraction, Agent 2 to the policy planner and template summary; answer still verified; `degraded_mode: true` |
| Loop bound hit | Verification fails ("run did not complete") → ESCALATE |
| Worker container killed mid-run | Run stays in `PROCESSING`; see RUNBOOK failure mode 3 |

## 4. Retrieval (L1, D2)
- Chunking: one chunk per numbered clause (e.g. 3.3.2) so a clause's amounts and its conditions stay
  together; headings are separate small chunks; clause ids are kept as metadata for exact lookup.
- Embeddings: `all-MiniLM-L6-v2` locally (baked into the image); deterministic hashing fallback if unavailable.
  The backend is recorded in the manifest and vectors are re-embedded if it changes.
- Index: FAISS via LangChain (inner product on normalised vectors).
- Why this addresses the difficulty: dense retrieval finds the amounts; reference following and the derived
  modifier search find the conditions (3.3.1 notice, 3.3.4 exemptions, 3.7 facilities) that a single
  similarity search misses.
- Idempotent pipeline (`python -m src.ingestion.pipeline`): content-hash chunk ids; only new/changed chunks
  are embedded; removed chunks dropped. Freshness and throughput exposed as metrics and at `/api/v1/index/freshness`.

## 5. Section 4.1 case
Agent trace: `corpus_search` → CAR 3.3.2 (amounts + "Where the passenger accepts an alternate flight … under
Clause 3.7 and 3.8") → `lookup_clause_by_id CAR-3.7` → `lookup_clause_by_id CAR-3.8` → derived search for
exemptions → decide → verify. Output: request-further-evidence, INR 0, meals and refreshments, citations
CAR-3.3.2 and CAR-3.7. The v1.0 baseline (single pass) does not read 3.7 and gets this case wrong.

## 6. Guardrails (L4)
Input: prompt-injection and toxicity patterns on the free-text field before any agent runs (quarantined).
Output: authority filter (the system cannot state that a payment was authorised or made) and toxicity filter.
The LLM never computes amounts; amounts come from the evidence-gated calculator.

## 7. Cost (L5)
Every model call records prompt/completion tokens and cost at call time, attributed to its step
(`agent_1_classification_extraction`, `agent_2_plan`, `agent_2_explain`) and summed per run
(`cost_breakdown` in each run record; `cost_per_100_runs_usd` in the evaluation report).
The largest driver is the repeated planner call (one per agent step); the whole-run cache removes all model
calls for repeated claims. Report your measured figures from `src.eval.runner` with the provider you use.

## 8. SLO (D6)
p95 of a complete run (submission → final result) ≤ 5.0 s, declared before measurement.
Measured with Locust against the llm-stub service; see README section 7.

## 9. Design decisions and rejected options
- **Deterministic amounts, model for planning/explanation**: amounts are legal facts; letting a model compute
  them risks fabricated entitlements. Rejected: model-generated amounts.
- **Evidence-gated rules**: rules can only use retrieved provisions, so retrieval quality decides correctness and
  every citation is verifiable. Rejected: hard-coded quotes (the original version), which made verification meaningless.
- **Redis for shared state**: rejected per-container files (caused INC-2026-11-10-02).
- **One image, several roles**: rejected separate images (doubled build size and crashed Docker Desktop, INC-2026-11-10-01).
- **LLM stub as a service**: rejected an in-process stub for load tests because it skips HTTP, timeouts and the breaker.
