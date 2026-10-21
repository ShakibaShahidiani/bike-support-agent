"""Code-level guardrails. Each check is a small pure function so it can be unit tested and explained."""
import re

from agent import config

# ---------- input ----------

def input_too_long(message: str) -> bool:
    return len(message) > config.MAX_INPUT_CHARS


CONFIRM_RE = re.compile(r"^\s*(yes|yes please|confirm|i confirm|yes,? (start|confirm) the return)\s*[.!]?\s*$", re.I)


def is_confirmation(message: str) -> bool:
    """Strict: only a short, explicit yes counts. 'yes but...' or 'maybe' do not."""
    return bool(CONFIRM_RE.match(message))


# ---------- safety escalation ----------

SAFETY_PATTERNS = [
    r"\bcrash(ed|es)?\b", r"\binjur(y|ed|ies)\b", r"\bhurt\b", r"\bfell off\b", r"\bhospital\b",
    r"\bcrack(ed|s)?\b.*\bframe\b", r"\bframe\b.*\b(crack(ed|s)?|broken|bent|damage[d]?)\b",
    r"\bbattery\b.*\b(smok\w*|swell\w*|swollen|fire|burn\w*|overheat\w*|leak\w*|hot)\b",
    r"\b(smok\w*|swell\w*|swollen|fire|burn\w*|overheat\w*|leak\w*)\b.*\bbattery\b",
]
SAFETY_RE = re.compile("|".join(SAFETY_PATTERNS), re.I)


def needs_safety_escalation(message: str) -> bool:
    return bool(SAFETY_RE.search(message))


ESCALATION_INSTRUCTION = (
    "SAFETY ESCALATION: The customer reported a possible crash, injury, frame damage or battery hazard. "
    "Reply in 2-3 sentences: show empathy, tell them to stop riding / stop using the battery, and say you are "
    "handing them over to a human support agent. Give NO technical advice, NO repair steps, NO numbers."
)
ESCALATION_FALLBACK = (
    "I'm sorry to hear that. For your safety, please stop riding the bike and don't use or charge the battery. "
    "I'm handing you over to a human support agent who will contact you as soon as possible."
)
REPAIR_VERBS_RE = re.compile(r"\b(tighten|loosen|adjust|replace|remove|unscrew|torque|charge it|reinstall|glue|tape)\b", re.I)
HANDOVER_RE = re.compile(r"\b(human|support agent|colleague|team|specialist)\b", re.I)


def escalation_reply_ok(answer: str) -> bool:
    return bool(HANDOVER_RE.search(answer)) and not REPAIR_VERBS_RE.search(answer) and not GROUNDED_NUMBER_RE.search(answer)


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


# ---------- PII redaction (for logs) ----------

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"\+?\d[\d\s().-]{7,}\d")


def redact_pii(text: str) -> str:
    return PHONE_RE.sub("[PHONE]", EMAIL_RE.sub("[EMAIL]", text))
