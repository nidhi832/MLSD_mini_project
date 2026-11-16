"""
LLM stub service (D6, A5): an OpenAI-compatible /v1/chat/completions endpoint that
returns fixed responses after a realistic delay.

Why: load tests must measure this system, not a free-tier provider's rate limits.
The agents call it over HTTP exactly as they call Groq or Gemini, so timeouts,
retries, the circuit breaker and per-step cost accounting are all exercised.

    STUB_DELAY_SECONDS   mean response delay (default 0.6 s)
    STUB_JITTER_SECONDS  +/- uniform jitter (default 0.2 s)
    STUB_FAILURE_RATE    share of requests answered with HTTP 503 (default 0)

Demo item 8: `docker compose stop llm-stub` makes the provider unavailable; the
circuit breaker opens after CIRCUIT_BREAKER_FAILURE_THRESHOLD failures and runs
take the degradation path. `docker compose start llm-stub` and the breaker
half-opens after CIRCUIT_BREAKER_RECOVERY_SECONDS and closes on success.
"""

import asyncio
import json
import os
import random

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="LLM stub service")

DELAY = float(os.getenv("STUB_DELAY_SECONDS", "0.6"))
JITTER = float(os.getenv("STUB_JITTER_SECONDS", "0.2"))
FAILURE_RATE = float(os.getenv("STUB_FAILURE_RATE", "0"))


@app.get("/health")
async def health():
    return {"status": "ok", "delay_seconds": DELAY, "failure_rate": FAILURE_RATE}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    await asyncio.sleep(max(0.0, DELAY + random.uniform(-JITTER, JITTER)))
    if FAILURE_RATE and random.random() < FAILURE_RATE:
        return JSONResponse({"error": {"message": "stub: simulated provider failure"}}, status_code=503)

    user = body["messages"][-1]["content"]
    if "Choose the NEXT action" in user:
        # Fixed answer; the agent's guards keep it reading references and modifiers before finishing.
        content = json.dumps({"tool": "finish", "args": {}, "reason": "stub planner"})
    elif "Extract ONLY" in user:
        content = json.dumps({"disruption_type": "unclear", "notice_hours_before_departure": None,
                              "took_alternate_flight": None, "alternate_flight_delay_hours": None,
                              "cause_mentioned": None})
    else:
        content = "The determination follows from the retrieved provisions cited below."

    prompt_tokens = sum(len(m.get("content", "").split()) for m in body["messages"]) + 20
    return {
        "id": "stub-completion",
        "object": "chat.completion",
        "model": body.get("model", "stub-model"),
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": max(1, len(content.split()))},
    }
