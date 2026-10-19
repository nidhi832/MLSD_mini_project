"""
Asynchronous Task Queue & Job Runner (A4).
Supports Redis Queue with graceful in-memory queue fallback for standalone local execution.
"""

import os
import json
import time
from typing import Dict, Any, Optional
from collections import deque
import threading


class TaskQueue:
    def __init__(
        self,
        queue_name: str = "claim_processing_queue",
        redis_host: str = "localhost",
        redis_port: int = 6379,
    ):
        self.queue_name = queue_name
        self.redis_client = None
        self._memory_queue = deque()
        self._lock = threading.Lock()

        # Attempt Redis connection
        try:
            import redis
            client = redis.Redis(
                host=os.getenv("REDIS_HOST", redis_host),
                port=int(os.getenv("REDIS_PORT", redis_port)),
                db=int(os.getenv("REDIS_DB", 0)),
                socket_connect_timeout=1.0,
                decode_responses=True,
            )
            client.ping()
            self.redis_client = client
        except Exception:
            self.redis_client = None

    def enqueue(self, task_data: Dict[str, Any]) -> str:
        payload = json.dumps(task_data)
        if self.redis_client:
            self.redis_client.rpush(self.queue_name, payload)
        else:
            with self._lock:
                self._memory_queue.append(payload)
        return task_data.get("run_id", "unknown")

    def dequeue(self, timeout_seconds: int = 2) -> Optional[Dict[str, Any]]:
        if self.redis_client:
            try:
                item = self.redis_client.blpop(self.queue_name, timeout=timeout_seconds)
                if item:
                    return json.loads(item[1])
            except Exception:
                pass
        else:
            with self._lock:
                if self._memory_queue:
                    raw = self._memory_queue.popleft()
                    return json.loads(raw)
            time.sleep(0.05)

        return None

    def length(self) -> int:
        if self.redis_client:
            try:
                return self.redis_client.llen(self.queue_name)
            except Exception:
                return 0
        with self._lock:
            return len(self._memory_queue)

    def is_using_redis(self) -> bool:
        return self.redis_client is not None
