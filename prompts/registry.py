"""
Prompt Registry for versioned prompt artifacts (L2).
"""

from typing import Dict, Any
from prompts import prompt_v1, prompt_v2


def get_prompts_by_version(version: str = "v2.0") -> Dict[str, str]:
    if version in ("v1", "v1.0"):
        return {
            "version": "v1.0",
            "classifier": prompt_v1.SYSTEM_PROMPT_CLASSIFIER_V1,
            "reasoner": prompt_v1.SYSTEM_PROMPT_REASONER_V1,
        }
    elif version in ("v2", "v2.0"):
        return {
            "version": "v2.0",
            "classifier": prompt_v2.SYSTEM_PROMPT_CLASSIFIER_V2,
            "reasoner": prompt_v2.SYSTEM_PROMPT_REASONER_V2,
        }
    else:
        raise ValueError(f"Unknown prompt version: {version}")
