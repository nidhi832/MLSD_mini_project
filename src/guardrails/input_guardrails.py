"""
Input Guardrails: Prompt Injection and Toxicity Defence (L4).
Protects downstream multi-agent reasoning from prompt injection, adversarial overrides, and toxic input.
"""

import re
from typing import Tuple, List, Optional


class InputGuardrailException(Exception):
    """Raised when an input security check fails."""
    pass


class InputGuardrails:
    # High-risk prompt injection and system override signatures
    INJECTION_PATTERNS = [
        r"(?i)ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
        r"(?i)system\s+prompt\s+override",
        r"(?i)disregard\s+(the\s+)?(rules|regulations|car|dgca)",
        r"(?i)you\s+are\s+now\s+(in\s+developer\s+mode|an\s+unrestricted|dan)",
        r"(?i)approve\s+(all\s+)?compensation\s+(immediately|now|without\s+verification)",
        r"(?i)pay\s+maximum\s+(compensation|amount)",
        r"(?i)bypass\s+(all\s+)?(verification|checks|guardrails)",
        r"(?i)output\s+(the\s+)?system\s+prompt",
        r"(?i)execute\s+(python|bash|sql|command)",
    ]

    # Explicit toxicity & profanity terms
    TOXICITY_PATTERNS = [
        r"(?i)\b(kill\s+yourself|die\s+in\s+a\s+fire|terrorist|bomb\s+threat|hijack)\b",
        r"(?i)\b(idiots|scumbags|bastards|bitch|asshole|fuck\s+you)\b",
    ]

    @classmethod
    def inspect_text(cls, text: str) -> Tuple[bool, Optional[str], str]:
        """
        Inspects text for injection and toxicity.
        Returns: (is_safe: bool, violation_type: Optional[str], details: str)
        """
        if not text:
            return True, None, "Clean"

        # 1. Prompt Injection Detection
        for pattern in cls.INJECTION_PATTERNS:
            match = re.search(pattern, text)
            if match:
                matched_phrase = match.group(0)
                return False, "PROMPT_INJECTION_DETECTED", f"Blocked adversarial instruction pattern: '{matched_phrase}'"

        # 2. Toxicity Detection
        for pattern in cls.TOXICITY_PATTERNS:
            match = re.search(pattern, text)
            if match:
                matched_phrase = match.group(0)
                return False, "TOXICITY_DETECTED", f"Blocked toxic/abusive language pattern: '{matched_phrase}'"

        # 3. Structural Delimiter Injection Check
        if text.count("```") > 4 or "<|im_start|>" in text or "<|endoftext|>" in text:
            return False, "PROMPT_INJECTION_DETECTED", "Blocked raw LLM control token sequence"

        return True, None, "Clean"
