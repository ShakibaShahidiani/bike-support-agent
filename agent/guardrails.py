"""Code-level guardrails. Each check is a small pure function so it can be unit tested and explained."""
import re

from agent import config

# ---------- input ----------

def input_too_long(message: str) -> bool:
    return len(message) > config.MAX_INPUT_CHARS


# ---------- grounding post-check ----------

GROUNDED_NUMBER_RE = re.compile(r"(€\s?\d+(?:[.,]\d+)?|\d+(?:[.,]\d+)?\s?(?:mm|wh|kg|v|km|days?|€|eur)\b)", re.I)
GROUNDING_FALLBACK = (
    "I'm not able to confirm those details from our records right now. "
    "Would you like me to hand you over to a human support agent?"
)


def ungrounded_numbers(answer: str, tool_results_text: str) -> list[str]:
    """Numbers with spec/policy units in the answer that don't appear in this turn's tool results."""
    source_numbers = set(re.findall(r"\d+(?:[.,]\d+)?", tool_results_text))
    missing = []
    for match in GROUNDED_NUMBER_RE.findall(answer):
        number = re.search(r"\d+(?:[.,]\d+)?", match).group()
        if number not in source_numbers:
            missing.append(match.strip())
    return missing


# ---------- model / API failure ----------

MODEL_ERROR_FALLBACK = (
    "Sorry, I'm having technical trouble right now. Please try again in a moment, "
    "or I can hand you over to a human support agent."
)


# ---------- PII redaction (for logs) ----------

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE_RE = re.compile(r"\+?\d[\d\s().-]{7,}\d")


def redact_pii(text: str) -> str:
    return PHONE_RE.sub("[PHONE]", EMAIL_RE.sub("[EMAIL]", text))
