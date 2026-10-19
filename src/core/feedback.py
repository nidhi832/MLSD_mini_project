"""
Feedback Collection & Aggregation Store (L6).
Allows human analysts to score live triage recommendations and chart quality over time.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class FeedbackSubmission(BaseModel):
    run_id: str
    is_satisfactory: bool
    rating: int = Field(ge=1, le=5, description="1 to 5 quality rating")
    reason: str = Field(min_length=3, max_length=1000)
    reviewed_by: Optional[str] = "airline_analyst"


class FeedbackStore:
    def __init__(self, storage_dir: str = "data/feedback"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.index_file = os.path.join(self.storage_dir, "feedback.jsonl")

    def record_feedback(self, submission: FeedbackSubmission) -> Dict[str, Any]:
        record = {
            "feedback_id": f"fb-{int(time.time()*1000)}",
            "run_id": submission.run_id,
            "is_satisfactory": submission.is_satisfactory,
            "rating": submission.rating,
            "reason": submission.reason,
            "reviewed_by": submission.reviewed_by,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        with open(self.index_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        return record

    def get_summary(self) -> Dict[str, Any]:
        if not os.path.exists(self.index_file):
            return {
                "total_reviews": 0,
                "satisfactory_count": 0,
                "unsatisfactory_count": 0,
                "satisfaction_rate": 0.0,
                "average_rating": 0.0,
                "recent_feedback": [],
            }

        records = []
        with open(self.index_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line.strip()))

        total = len(records)
        sat_count = sum(1 for r in records if r["is_satisfactory"])
        avg_rating = sum(r["rating"] for r in records) / total if total > 0 else 0.0

        return {
            "total_reviews": total,
            "satisfactory_count": sat_count,
            "unsatisfactory_count": total - sat_count,
            "satisfaction_rate": round(sat_count / total, 4) if total > 0 else 0.0,
            "average_rating": round(avg_rating, 2),
            "recent_feedback": records[-10:],
        }
