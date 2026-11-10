"""
Resilience & Cost Tracking Layer (A5, L5).
Includes:
1. Circuit Breaker for LLM provider failures with graceful degradation
2. Bounded retry with exponential backoff
3. Per-step & per-run token and cost tracking with cost-per-100-runs projection
"""

import time
import math
import random
from enum import Enum
from typing import Dict, Any, List, Optional, Callable


class CircuitState(str, Enum):
    CLOSED = "CLOSED"        # Normal operation: requests pass through
    OPEN = "OPEN"            # Tripped: immediate failover to degradation path
    HALF_OPEN = "HALF_OPEN"  # Testing recovery


class CircuitBreakerOpenException(Exception):
    """Raised when request is rejected because circuit is OPEN."""
    pass


class CircuitBreaker:
    """
    Circuit Breaker implementing fail-fast and graceful degradation (A5).

    State is shared through Redis (key circuit:<name>) so that every container
    that calls the provider sees the same breaker: tripping it through the API
    (demo item 8) also stops the agent workers, and failures seen by one worker
    count for all of them. Without Redis the state is per process.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout_seconds: float = 30.0,
        name: str = "llm_provider",
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_state_change = time.time()
        self.last_failure_time = 0.0
        self.total_trips = 0

    # -- shared state -------------------------------------------------------
    def _key(self) -> str:
        return f"circuit:{self.name}"

    def _load(self):
        try:
            from src.core.shared_state import get_redis
            client = get_redis()
            if client is None:
                return
            data = client.hgetall(self._key())
            if data:
                self.state = CircuitState(data.get("state", "CLOSED"))
                self.failure_count = int(data.get("failure_count", 0))
                self.last_state_change = float(data.get("last_state_change", time.time()))
                self.last_failure_time = float(data.get("last_failure_time", 0.0))
                self.total_trips = int(data.get("total_trips", 0))
        except Exception:
            pass

    def _save(self):
        try:
            from src.core.shared_state import get_redis
            client = get_redis()
            if client is None:
                return
            client.hset(self._key(), mapping={
                "state": self.state.value,
                "failure_count": self.failure_count,
                "last_state_change": self.last_state_change,
                "last_failure_time": self.last_failure_time,
                "total_trips": self.total_trips,
            })
        except Exception:
            pass

    def can_execute(self) -> bool:
        self._load()
        if self.state == CircuitState.CLOSED:
            return True

        now = time.time()
        if self.state == CircuitState.OPEN:
            if now - self.last_state_change >= self.recovery_timeout_seconds:
                self.state = CircuitState.HALF_OPEN
                self.last_state_change = now
                self._save()
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            return True

        return False

    def record_success(self):
        self._load()
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.last_state_change = time.time()
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0
        self._save()

    def record_failure(self):
        self._load()
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.CLOSED and self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.last_state_change = time.time()
            self.total_trips += 1
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.last_state_change = time.time()
            self.total_trips += 1
        self._save()

    def trip_manually(self):
        """Allows testing and demonstration of circuit breaker trip (Demo Item 8)."""
        self._load()
        self.state = CircuitState.OPEN
        self.last_state_change = time.time()
        self.total_trips += 1
        self._save()

    def reset_manually(self):
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_state_change = time.time()
        self._save()

    def get_status(self) -> Dict[str, Any]:
        self._load()
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "threshold": self.failure_threshold,
            "total_trips": self.total_trips,
            "seconds_since_last_change": round(time.time() - self.last_state_change, 2),
        }


class StepCostRecord:
    def __init__(self, step_name: str, prompt_tokens: int, completion_tokens: int, duration_seconds: float):
        self.step_name = step_name
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.duration_seconds = duration_seconds
        # Pricing model: $0.15 per 1M input tokens, $0.60 per 1M output tokens (standard lightweight LLM)
        self.cost_usd = (prompt_tokens * 0.15 / 1_000_000) + (completion_tokens * 0.60 / 1_000_000)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_name": self.step_name,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "duration_seconds": round(self.duration_seconds, 4),
        }


class CostTracker:
    """Tracks token usage and cost attributed per step and per run (L5)."""

    def __init__(self):
        self.steps: List[StepCostRecord] = []

    def record_step(self, step_name: str, prompt_tokens: int, completion_tokens: int, duration_seconds: float):
        record = StepCostRecord(step_name, prompt_tokens, completion_tokens, duration_seconds)
        self.steps.append(record)

    def get_totals(self) -> Dict[str, Any]:
        total_prompt = sum(s.prompt_tokens for s in self.steps)
        total_completion = sum(s.completion_tokens for s in self.steps)
        total_cost = sum(s.cost_usd for s in self.steps)

        # Identify largest single cost driver
        driver = max(self.steps, key=lambda s: s.cost_usd) if self.steps else None
        largest_driver_name = driver.step_name if driver else "none"
        largest_driver_cost = round(driver.cost_usd, 6) if driver else 0.0

        return {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "total_cost_usd": round(total_cost, 6),
            "cost_per_100_runs_usd": round(total_cost * 100, 4),
            "largest_single_cost_driver": largest_driver_name,
            "largest_driver_cost_usd": largest_driver_cost,
            "steps": [s.to_dict() for s in self.steps],
        }


def execute_with_retry(
    func: Callable,
    max_retries: int = 3,
    initial_delay: float = 0.5,
    backoff_factor: float = 2.0,
    circuit_breaker: Optional[CircuitBreaker] = None,
    timeout_seconds: float = 15.0,
    on_retry: Optional[Callable[[int], None]] = None,
    *args,
    **kwargs,
) -> Any:
    """Executes a callable with bounded exponential backoff and circuit breaker protection (A5)."""
    if circuit_breaker and not circuit_breaker.can_execute():
        raise CircuitBreakerOpenException(
            f"Circuit breaker '{circuit_breaker.name}' is OPEN. Provider unavailable; entering graceful degradation."
        )

    delay = initial_delay
    last_err = None

    for attempt in range(max_retries):
        if attempt > 0 and on_retry:
            on_retry(attempt)
        if circuit_breaker and attempt > 0 and not circuit_breaker.can_execute():
            raise CircuitBreakerOpenException(
                f"Circuit breaker '{circuit_breaker.name}' opened during retries; entering graceful degradation."
            )
        try:
            start_t = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_t
            if elapsed > timeout_seconds:
                raise TimeoutError(f"Call exceeded timeout limit of {timeout_seconds}s (took {elapsed:.2f}s)")

            if circuit_breaker:
                circuit_breaker.record_success()
            return result
        except Exception as e:
            last_err = e
            if circuit_breaker:
                circuit_breaker.record_failure()

            if attempt == max_retries - 1:
                break

            jitter = random.uniform(0.8, 1.2)
            time.sleep(delay * jitter)
            delay *= backoff_factor

    raise last_err
