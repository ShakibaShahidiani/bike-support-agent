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


CANCEL_RE = re.compile(r"^\s*(no|nope|cancel|no,? (thanks|thank you)|cancel (it|the return)|don'?t)\s*[.!]?\s*$", re.I)


def is_cancellation(message: str) -> bool:
    return bool(CANCEL_RE.match(message))


RETURN_INTENT_RE = re.compile(r"\breturn(ing)?\b|\bsend (it|this|them|the bike) back\b", re.I)


def has_return_intent(message: str) -> bool:
    return bool(RETURN_INTENT_RE.search(message))


# Words that make a message a follow-up about the pending return (keeps it pending).
PENDING_RELATED_RE = re.compile(
    r"^\s*yes\b|\b(return|refund|ship(ping)?|label|pick ?up|cost|pay|policy|days?|window|box|"
    r"condition|km|inspect(ion)?|confirm)\b", re.I)
ORDER_ID_IN_TEXT_RE = re.compile(r"\bCN-\d{4}\b", re.I)


def is_related_to_pending(message: str, order_id: str) -> bool:
    """A question or 'yes but...' about the pending return keeps it pending; a new request does not."""
    other_orders = {o.upper() for o in ORDER_ID_IN_TEXT_RE.findall(message)} - {order_id}
    if other_orders:
        return False
    return order_id.lower() in message.lower() or bool(PENDING_RELATED_RE.search(message))


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
