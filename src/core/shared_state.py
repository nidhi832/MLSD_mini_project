"""
Shared state backend (A4, A7, D7).

The API, the classifier agent and the reasoning agent run in separate containers,
so anything they must all see (run records, the whole-run cache, circuit breaker
state) lives in Redis. When Redis is not reachable (unit tests, local runs without
Docker) every caller falls back to process memory / local files.
"""

import os
import threading
from typing import Optional

_client = None
_checked = False
_lock = threading.Lock()


def get_redis():
    """Return a connected Redis client, or None when Redis is unavailable."""
    global _client, _checked
    if _checked:
        return _client
    with _lock:
        if _checked:
            return _client
        if os.getenv("DISABLE_REDIS", "").lower() in ("1", "true", "yes"):
            _checked = True
            return None
        try:
            import redis

            client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                db=int(os.getenv("REDIS_DB", "0")),
                socket_connect_timeout=1.0,
                socket_timeout=5.0,
                decode_responses=True,
            )
            client.ping()
            _client = client
        except Exception:
            _client = None
        _checked = True
        return _client


def reset_for_tests(client: Optional[object] = None):
    """Force a specific client (or None) — used by the test suite."""
    global _client, _checked
    with _lock:
        _client = client
        _checked = True
