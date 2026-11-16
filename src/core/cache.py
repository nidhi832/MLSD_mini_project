"""
Caching Layer (D7).

Whole-run cache keyed by a hash of every claim field that can change the outcome.
Justification: an identical claim (same facts, same prompt version) must get the
identical determination, and a cache hit skips every LLM call, so it removes the
largest cost driver and the largest latency component of a run.

Backed by Redis so all scaled agent-worker replicas share one cache; falls back
to process memory without Redis. Disable with CACHE_ENABLED=false to measure the
effect of caching on p95 latency and cost per run.
"""

import hashlib
import json
import os
import time
from typing import Dict, Any, Optional

from src.core.shared_state import get_redis

# Fields that identify the submission rather than the facts; they do not affect the outcome.
_IDENTITY_FIELDS = {"claim_id", "passenger_name", "contact_email", "contact_phone"}


class ClaimCache:
    KEY_PREFIX = "cache:run:"

    def __init__(self, default_ttl_seconds: Optional[int] = None):
        self.default_ttl = default_ttl_seconds or int(os.getenv("RESULT_CACHE_TTL", "86400"))
        self.enabled = os.getenv("CACHE_ENABLED", "true").lower() not in ("0", "false", "no")
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0

    def compute_claim_hash(self, claim_data: Dict[str, Any], prompt_version: str = "v2.0") -> str:
        canonical = {k: v for k, v in claim_data.items() if k not in _IDENTITY_FIELDS}
        canonical["__prompt_version"] = prompt_version
        canonical["__llm_model"] = os.getenv("LLM_MODEL", "stub")
        key_str = json.dumps(canonical, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode("utf-8")).hexdigest()

    def get_run(self, claim_hash: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        client = get_redis()
        if client is not None:
            try:
                raw = client.get(self.KEY_PREFIX + claim_hash)
                if raw:
                    self.hits += 1
                    return json.loads(raw)
                self.misses += 1
                return None
            except Exception:
                pass

        entry = self._memory_cache.get(claim_hash)
        if entry and time.time() < entry["expires_at"]:
            self.hits += 1
            return entry["result"]
        self._memory_cache.pop(claim_hash, None)
        self.misses += 1
        return None

    def set_run(self, claim_hash: str, result: Dict[str, Any], ttl_seconds: Optional[int] = None):
        if not self.enabled:
            return
        ttl = ttl_seconds or self.default_ttl
        client = get_redis()
        if client is not None:
            try:
                client.set(self.KEY_PREFIX + claim_hash, json.dumps(result, default=str), ex=ttl)
                return
            except Exception:
                pass
        self._memory_cache[claim_hash] = {
            "result": result,
            "expires_at": time.time() + ttl,
            "cached_at": time.time(),
        }

    def get_stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            "enabled": self.enabled,
            "hits": self.hits,
            "misses": self.misses,
            "total_lookups": total,
            "hit_ratio": round(self.hits / total, 4) if total else 0.0,
            "entries_count": len(self._memory_cache),
        }
