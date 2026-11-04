"""Unit tests for the code guardrails (no LLM calls)."""
from langchain_core.documents import Document
from langgraph.types import Command

from agent import guardrails as g
from agent.tracing import _mask


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


def test_pii_redaction():
    assert g.redact_pii("mail me at jan.doe@example.com or +31 6 1234 5678") == "mail me at [EMAIL] or [PHONE]"


def test_input_length():
    assert g.input_too_long("x" * 2001)
    assert not g.input_too_long("x" * 2000)


def test_trace_mask_redacts_nested_data_and_objects():
    masked = _mask({"a": ["mail me at a.b@example.com"], "n": 3, "other": object, "cmd": Command(update={"message": "a.b@example.com"})})
    assert "a.b@example.com" not in str(masked)
    assert masked["n"] == 3


def test_trace_shows_decisions_not_framework_objects():
    command = Command(goto="confirm", update={"history": [{"role": "user", "content": "x"}], "order_id": "CN-1042", "reason": None})
    assert _mask(command) == {"next": "confirm", "changes": {"order_id": "CN-1042"}}
    assert _mask({"message": "hi", "history": [1], "year": None, "model": "Aero R1"}) == {"message": "hi", "model": "Aero R1"}
    assert _mask({"role": "assistant", "content": '{"intent":"return"}', "additional_kwargs": {}}) == {"intent": "return"}
    assert _mask(Document(page_content="text", metadata={"source": "a.md"})) == {"source": "a.md", "content": "text"}
    assert _mask(Command(resume="yes")) == {"resume_with": "yes"}
    assert _mask({"id": "x", "page_content": "text", "metadata": {"source": "a.md"}}) == {"source": "a.md", "content": "text"}
