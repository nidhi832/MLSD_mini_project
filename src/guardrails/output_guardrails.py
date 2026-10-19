"""
Output Guardrails: Authority and Toxicity Filtering (L4, Section 2).
Ensures the system NEVER issues an unauthorized payment execution or binding commitment,
and ensures output neutrality and toxicity-free responses.
"""

import re
from typing import Tuple, Optional


class OutputGuardrails:
    # Patterns where model erroneously attempts to bind the company or execute bank transfers
    UNAUTHORIZED_AUTHORITY_PATTERNS = [
        r"(?i)\b(i\s+have\s+.*(authorized|credited|refunded|disbursed).*account)\b",
        r"(?i)\b(bank\s+transfer\s+of\s+inr\s+.*has\s+been\s+initiated)\b",
        r"(?i)\b(your\s+payout\s+is\s+approved\s+and\s+disbursed)\b",
        r"(?i)\b(we\s+hereby\s+guarantee\s+immediate\s+cash\s+transfer)\b",
        r"(?i)\b(authorized\s+payment\s+of\s+inr)\b",
    ]

    TOXICITY_PATTERNS = [
        r"(?i)\b(idiot|fool|shut\s+up|stupid|worthless)\b",
    ]

    @classmethod
    def filter_output(cls, reasoning_text: str) -> Tuple[bool, Optional[str], str]:
        """
        Validates model output against authority limits and neutrality.
        Returns: (is_valid: bool, violation_type: Optional[str], sanitized_text: str)
        """
        for pattern in cls.UNAUTHORIZED_AUTHORITY_PATTERNS:
            if re.search(pattern, reasoning_text):
                # Replace with authorized advisory language
                sanitized = re.sub(
                    pattern,
                    "This triage recommendation has been flagged for airline customer relations review",
                    reasoning_text,
                )
                return False, "UNAUTHORIZED_DETERMINATION_ATTEMPT", sanitized

        for pattern in cls.TOXICITY_PATTERNS:
            if re.search(pattern, reasoning_text):
                return False, "OUTPUT_TOXICITY_DETECTED", "Response suppressed due to policy violation."

        return True, None, reasoning_text
