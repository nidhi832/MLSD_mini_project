"""
Quarantine Store for rejected/malformed claim requests (F2, D1).
Ensures malformed requests never trigger an agent run and are persisted with detailed field errors.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class QuarantinedRecord(BaseModel):
    quarantine_id: str
    correlation_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    raw_payload: Dict[str, Any]
    client_ip: Optional[str] = None
    validation_errors: List[Dict[str, Any]]
    rejection_reason: str


class QuarantineStore:
    def __init__(self, storage_dir: str = "data/quarantine"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.index_file = os.path.join(self.storage_dir, "quarantine_index.jsonl")

    def save_rejected_claim(
        self,
        quarantine_id: str,
        correlation_id: str,
        raw_payload: Dict[str, Any],
        validation_errors: List[Dict[str, Any]],
        rejection_reason: str,
        client_ip: Optional[str] = None,
    ) -> QuarantinedRecord:
        record = QuarantinedRecord(
            quarantine_id=quarantine_id,
            correlation_id=correlation_id,
            raw_payload=raw_payload,
            client_ip=client_ip,
            validation_errors=validation_errors,
            rejection_reason=rejection_reason,
        )

        # Write to JSONL append log
        with open(self.index_file, "a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")

        # Also write individual file for inspection
        single_file = os.path.join(self.storage_dir, f"{quarantine_id}.json")
        with open(single_file, "w", encoding="utf-8") as f:
            f.write(record.model_dump_json(indent=2))

        return record

    def list_records(self, limit: int = 50) -> List[Dict[str, Any]]:
        if not os.path.exists(self.index_file):
            return []
        records = []
        with open(self.index_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line.strip()))
        return records[-limit:]

    def get_record(self, quarantine_id: str) -> Optional[Dict[str, Any]]:
        single_file = os.path.join(self.storage_dir, f"{quarantine_id}.json")
        if os.path.exists(single_file):
            with open(single_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
