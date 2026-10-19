"""
Client Rate Limiting Layer (D7).
Token Bucket Algorithm per client IP to prevent budget exhaustion from rogue loops.
"""

import time
from typing import Dict, Tuple


class TokenBucketRateLimiter:
    def __init__(self, rate_per_second: float = 10.0, burst_capacity: float = 20.0):
        self.rate = rate_per_second
        self.capacity = burst_capacity
        self.buckets: Dict[str, Dict[str, float]] = {}

    def is_allowed(self, client_id: str) -> Tuple[bool, float]:
        """
        Returns: (allowed: bool, remaining_tokens: float)
        """
        now = time.time()
        bucket = self.buckets.get(client_id)

        if bucket is None:
            bucket = {"tokens": self.capacity, "last_updated": now}
            self.buckets[client_id] = bucket

        # Replenish tokens based on elapsed time
        elapsed = now - bucket["last_updated"]
        bucket["tokens"] = min(self.capacity, bucket["tokens"] + elapsed * self.rate)
        bucket["last_updated"] = now

        if bucket["tokens"] >= 1.0:
            bucket["tokens"] -= 1.0
            return True, bucket["tokens"]
        else:
            return False, bucket["tokens"]
