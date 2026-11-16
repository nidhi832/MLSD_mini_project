"""
LLM Provider Abstraction Layer (D4, D6, A5, L5).
Supports:
1. 'stub' provider: High-fidelity, deterministic simulated reasoning with configurable realistic delay (required by D6 for Locust load testing without provider rate limits).
2. Any OpenAI-compatible chat API, selected purely by configuration (D4):
   LLM_PROVIDER=stub-service -> http://llm-stub:8100/v1 (the Compose LLM stub, for load tests / demos)
   LLM_PROVIDER=groq   -> https://api.groq.com/openai/v1           (free tier)
   LLM_PROVIDER=gemini -> https://generativelanguage.googleapis.com/v1beta/openai (free tier)
   LLM_PROVIDER=ollama -> http://host.docker.internal:11434/v1     (local)
   LLM_PROVIDER=openai -> LLM_API_BASE (default https://api.openai.com/v1)
   No model name, key or endpoint is hard-coded: they come from LLM_MODEL,
   LLM_API_KEY and LLM_API_BASE at runtime.
3. Fully integrated with CircuitBreaker, CostTracker, and timeouts.
"""

import os
import time
import json
from typing import Dict, Any, List, Optional
from src.core.resilience import CircuitBreaker, CostTracker, execute_with_retry

OPENAI_COMPATIBLE_BASES = {
    "stub-service": "http://llm-stub:8100/v1",
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "ollama": "http://host.docker.internal:11434/v1",
    "openai": "https://api.openai.com/v1",
}


class LLMClient:
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
        cost_tracker: Optional[CostTracker] = None,
        stub_latency_seconds: float = 0.05,
    ):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "stub")).split("#")[0].strip().lower() or "stub"
        self.model = model or os.getenv("LLM_MODEL", "").strip() or "stub-deterministic-v1"
        self.circuit_breaker = circuit_breaker or CircuitBreaker(
            name="llm_provider",
            failure_threshold=int(os.getenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "3")),
            recovery_timeout_seconds=float(os.getenv("CIRCUIT_BREAKER_RECOVERY_SECONDS", "30")),
        )
        self.cost_tracker = cost_tracker or CostTracker()
        env_delay = os.getenv("STUB_LLM_DELAY_SECONDS")
        self.stub_latency_seconds = float(env_delay) if env_delay else stub_latency_seconds
        self.timeout_seconds = float(os.getenv("LLM_TIMEOUT_SECONDS", "15"))
        self.on_retry = None  # optional callback(attempt) for retry metrics

    def for_run(self) -> "LLMClient":
        """A view of this client with its own per-run cost tracker, sharing the circuit breaker (L5, A5)."""
        clone = LLMClient.__new__(LLMClient)
        clone.__dict__.update(self.__dict__)
        clone.cost_tracker = CostTracker()
        return clone

    @property
    def is_stub(self) -> bool:
        return self.provider == "stub"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        step_name: str = "llm_call",
        temperature: float = 0.0,
        timeout_seconds: Optional[float] = None,
        json_mode: bool = False,
    ) -> str:
        """
        Executes a call to the LLM backend through circuit breaker and retry.
        Tracks prompt tokens, completion tokens, latency, and cost per step.
        """
        start_t = time.time()
        timeout_seconds = timeout_seconds or self.timeout_seconds

        def _call_backend():
            if self.provider == "stub":
                if self.stub_latency_seconds > 0:
                    time.sleep(self.stub_latency_seconds)
                # Compute token counts based on standard word ratio
                p_tokens = len(system_prompt.split()) + len(user_prompt.split()) + 20
                c_tokens = 85
                response = f"[STUB-RESPONSE] Evaluated with model={self.model}"
                return response, p_tokens, c_tokens

            elif self.provider in OPENAI_COMPATIBLE_BASES:
                import httpx
                api_key = os.getenv("LLM_API_KEY", "").strip()
                api_base = (os.getenv("LLM_API_BASE", "").strip()
                            or OPENAI_COMPATIBLE_BASES[self.provider]).rstrip("/")
                if not api_key and self.provider not in ("ollama", "stub-service"):
                    raise ValueError("LLM_API_KEY environment variable not configured.")

                body = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": temperature,
                }
                if json_mode:
                    body["response_format"] = {"type": "json_object"}
                headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
                resp = httpx.post(f"{api_base}/chat/completions", headers=headers, json=body, timeout=timeout_seconds)
                if resp.status_code == 400 and json_mode:
                    # Some models reject response_format; ask again without it (JSON is still parsed leniently).
                    body.pop("response_format", None)
                    resp = httpx.post(f"{api_base}/chat/completions", headers=headers, json=body, timeout=timeout_seconds)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"].get("content") or ""
                p_tokens = data.get("usage", {}).get("prompt_tokens", len(user_prompt.split()))
                c_tokens = data.get("usage", {}).get("completion_tokens", len(content.split()))
                return content, p_tokens, c_tokens

            else:
                raise ValueError(f"Unsupported LLM provider: {self.provider}")

        response_text, prompt_tokens, completion_tokens = execute_with_retry(
            _call_backend,
            max_retries=3,
            initial_delay=0.2,
            circuit_breaker=self.circuit_breaker,
            timeout_seconds=timeout_seconds,
            on_retry=self.on_retry,
        )

        duration = time.time() - start_t
        self.cost_tracker.record_step(
            step_name=step_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration_seconds=duration,
        )

        return response_text

    def generate_json(self, system_prompt: str, user_prompt: str, step_name: str) -> Optional[Dict[str, Any]]:
        """Call the model asking for a JSON object; returns None if the reply is not valid JSON."""
        text = self.generate(system_prompt, user_prompt, step_name=step_name, json_mode=True)
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text[text.find("{"):]
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            return None
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
