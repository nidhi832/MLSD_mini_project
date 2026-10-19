"""
Caching Layer (D7).
Provides:
1. Whole-run caching (keyed by deterministic claim hash)
2. Retrieval and embedding caching
Controls cost and reduces p95 latency.
"""

import hashlib
import json
import time
from typing import Dict, Any, Optional


class ClaimCache:
    def __init__(self, default_ttl_seconds: int = 86400):
        self.default_ttl = default_ttl_seconds
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0

    def compute_claim_hash(self, claim_data: Dict[str, Any]) -> str:
        # Canonical hash of normalized core attributes
        key_str = f"{claim_data.get('pnr')}:{claim_data.get('flight_number')}:{claim_data.get('disruption_type')}:{claim_data.get('accepted_alternate_flight')}:{claim_data.get('block_time_hours')}"
        return hashlib.sha256(key_str.encode("utf-8")).hexdigest()

    def get_run(self, claim_hash: str) -> Optional[Dict[str, Any]]:
        entry = self._memory_cache.get(claim_hash)
        if entry:
            if time.time() < entry["expires_at"]:
                self.hits += 1
                return entry["result"]
            else:
                del self._memory_cache[claim_hash]
        self.misses += 1
        return None

    def set_run(self, claim_hash: str, result: Dict[str, Any], ttl_seconds: Optional[int] = None):
        ttl = ttl_seconds or self.default_ttl
        self._memory_cache[claim_hash] = {
            "result": result,
            "expires_at": time.time() + ttl,
            "cached_at": time.time(),
        }

    def get_stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        hit_ratio = round((self.hits / total), 4) if total > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_lookups": total,
            "hit_ratio": hit_ratio,
            "entries_count": len(self._memory_cache),
        }
