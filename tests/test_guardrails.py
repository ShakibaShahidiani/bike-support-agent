"""Unit tests for the code guardrails (no LLM calls)."""
import pytest

from agent import guardrails as g


@pytest.mark.parametrize("msg", ["yes", "Yes.", "confirm", "yes, start the return", "YES please"])
def test_confirmation_accepted(msg):
    assert g.is_confirmation(msg)


@pytest.mark.parametrize("msg", ["yes but what about shipping?", "maybe", "no", "sure, why not", "ok"])
def test_confirmation_rejected(msg):
    assert not g.is_confirmation(msg)


@pytest.mark.parametrize("msg", [
    "I crashed my Strada yesterday",
    "There's a crack in my frame near the bottom bracket",
    "my battery is swollen",
    "smoke came out of the battery",
    "I got injured when the fork snapped",
])
def test_safety_escalation_triggers(msg):
    assert g.needs_safety_escalation(msg)


@pytest.mark.parametrize("msg", ["how big is the Ridge E battery?", "how do I store my battery in winter?",
                                 "what frame material is the Aero R1?"])
def test_safety_escalation_ignores_normal_questions(msg):
    assert not g.needs_safety_escalation(msg)


def test_grounding_accepts_numbers_from_tools():
    tool_text = '{"max_rotor_front_mm": 180}'
    assert g.ungrounded_numbers("The 2025 model takes a 180mm rotor.", tool_text) == []


def test_grounding_flags_invented_numbers():
    tool_text = '{"max_rotor_front_mm": 180}'
    assert g.ungrounded_numbers("It takes 203 mm rotors and returns within 14 days.", tool_text) == ["203 mm", "14 days"]


def test_grounding_checks_euro_amounts():
    assert g.ungrounded_numbers("Return shipping costs €49.", "free return shipping") == ["€49"]


def test_grounding_ignores_years_and_order_ids():
    assert g.ungrounded_numbers("Your order CN-1042 for the 2025 Strada GX.", "") == []


def test_escalation_reply_check():
    assert g.escalation_reply_ok("I'm sorry. Please stop riding; I'm handing you to a human support agent.")
    assert not g.escalation_reply_ok("Tighten the axle to 12 Nm and contact a human.")
    assert not g.escalation_reply_ok("Please stop riding.")  # no handover


def test_pii_redaction():
    assert g.redact_pii("mail me at jan.doe@example.com or +31 6 1234 5678") == "mail me at [EMAIL] or [PHONE]"


def test_input_length():
    assert g.input_too_long("x" * 2001)
    assert not g.input_too_long("x" * 2000)


@pytest.mark.parametrize("msg", ["no", "Cancel", "no thanks", "cancel the return"])
def test_cancellation(msg):
    assert g.is_cancellation(msg)


@pytest.mark.parametrize("msg", ["no idea how shipping works?", "nothing else"])
def test_not_cancellation(msg):
    assert not g.is_cancellation(msg)


@pytest.mark.parametrize("msg,related", [
    ("who pays the shipping?", True), ("yes but how long does it take?", True),
    ("what about CN-1042?", True), ("where is CN-1044?", False),
    ("what's the rotor size on a 2025 Strada GX?", False),
])
def test_related_to_pending(msg, related):
    assert g.is_related_to_pending(msg, "CN-1042") is related


def test_return_intent():
    assert g.has_return_intent("I want to return my bike")
    assert not g.has_return_intent("where is my order?")
