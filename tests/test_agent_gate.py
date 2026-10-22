"""End-to-end test of the agent loop's confirmation gate with a scripted fake model (no API calls).

The fake model always tries to start a return. The gate must still only execute after an explicit 'yes'.
"""
import json
from datetime import date
from types import SimpleNamespace

import pytest

from agent import agent, config, session, tools


def _response(content=None, tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                           usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))


def _tool_call(name, args):
    return SimpleNamespace(id="call_1", function=SimpleNamespace(name=name, arguments=json.dumps(args)))


@pytest.fixture
def fake_model(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(tools, "date", SimpleNamespace(today=lambda: date(2026, 10, 21), fromisoformat=date.fromisoformat))
    tools.RETURNS.clear()
    session.SESSIONS.clear()

    def completion(model, messages, tools=None, **kwargs):
        # With tools: request start_return unless a tool result is already in the history.
        if tools and messages[-1]["role"] != "tool":
            return _response(tool_calls=[_tool_call("start_return", {"order_id": "CN-1042", "reason": "too small"})])
        return _response(content="Done.")

    monkeypatch.setattr(agent.litellm, "completion", completion)
    return tmp_path


def test_return_needs_explicit_confirmation(fake_model):
    first = agent.handle_turn("s", "I want to return order CN-1042, it's too small")
    assert first["pending_confirmation"] is True
    assert tools.RETURNS == {}

    second = agent.handle_turn("s", "yes")
    assert "return_executed" in second["flags"]
    assert list(tools.RETURNS) == ["CN-1042"]


def _pending(sid="s"):
    return session.get_session(sid).pending_action


def test_question_keeps_pending_then_yes_executes(fake_model):
    agent.handle_turn("s", "I want to return order CN-1042")
    out = agent.handle_turn("s", "hmm, what does shipping cost?")
    assert "pending_return_kept" in out["flags"]
    assert out["pending_confirmation"] is True
    assert agent.handle_turn("s", "yes")["flags"][0] == "return_executed"


def test_yes_but_keeps_pending(fake_model):
    agent.handle_turn("s", "I want to return order CN-1042")
    out = agent.handle_turn("s", "yes but first tell me who pays shipping")
    assert "pending_return_kept" in out["flags"]
    assert tools.RETURNS == {}


@pytest.mark.parametrize("msg", ["no", "cancel", "No thanks."])
def test_explicit_no_cancels_without_model_call(fake_model, msg):
    agent.handle_turn("s", "I want to return order CN-1042")
    out = agent.handle_turn("s", msg)
    assert "pending_return_cancelled" in out["flags"]
    assert _pending() is None
    line = json.loads((fake_model / "turns.jsonl").read_text().splitlines()[-1])
    assert line["latency_ms"]["model_calls"] == []


def test_unrelated_request_cancels_and_cannot_recreate(fake_model):
    agent.handle_turn("s", "I want to return order CN-1042")
    # the fake model tries start_return again, but this message doesn't ask for a return
    out = agent.handle_turn("s", "What's the max rotor on a 2025 Strada GX?")
    assert "pending_return_cancelled" in out["flags"]
    assert _pending() is None


def test_other_order_id_is_unrelated(fake_model):
    agent.handle_turn("s", "I want to return order CN-1042")
    out = agent.handle_turn("s", "and where is CN-1044?")
    assert "pending_return_cancelled" in out["flags"]


def test_pending_expires_after_two_follow_up_turns(fake_model):
    agent.handle_turn("s", "I want to return order CN-1042")
    agent.handle_turn("s", "who pays shipping?")
    agent.handle_turn("s", "is there a km limit?")
    out = agent.handle_turn("s", "yes")
    assert "pending_return_expired" in out["flags"]
    assert "expired" in out["answer"]
    assert tools.RETURNS == {}


def test_start_return_blocked_when_not_asked(fake_model):
    out = agent.handle_turn("s", "What's the battery on the Ridge E 2025?")
    assert out["tools_called"][0]["name"] == "start_return"  # the model tried...
    assert _pending() is None  # ...but code refused


def test_follow_up_answer_without_return_word_is_allowed(fake_model, monkeypatch):
    calls = iter([
        _response(tool_calls=[_tool_call("start_return", {"order_id": "CN-1042"})]),  # no reason yet
        _response(content="Why do you want to return it?"),
        _response(tool_calls=[_tool_call("start_return", {"order_id": "CN-1042", "reason": "too small"})]),
        _response(content="Reply yes to confirm."),
    ])
    monkeypatch.setattr(agent.litellm, "completion", lambda **kw: next(calls))
    agent.handle_turn("s", "I want to return CN-1042")
    agent.handle_turn("s", "it's too small")  # no 'return' word, but answers the follow-up
    assert _pending()["reason"] == "too small"


def test_turn_is_logged(fake_model):
    agent.handle_turn("s", "return CN-1042 please, mail me at a@b.com")
    line = json.loads((fake_model / "turns.jsonl").read_text().splitlines()[0])
    assert line["tools_called"][0]["name"] == "start_return"
    assert "[EMAIL]" in line["user_message"]
    assert "[PHONE]" not in line["timestamp"]
    assert line["input_tokens"] == 20


def test_confirmation_reply_is_templated_without_model_call(fake_model):
    agent.handle_turn("s", "Return CN-1042, too small")
    out = agent.handle_turn("s", "yes")
    assert "templated_reply" in out["flags"]
    assert "RT-0001" in out["answer"]
    line = json.loads((fake_model / "turns.jsonl").read_text().splitlines()[-1])
    assert line["latency_ms"]["model_calls"] == []  # no model call on the confirmation turn


def test_answer_is_redacted_in_log(fake_model, monkeypatch):
    monkeypatch.setattr(agent.litellm, "completion",
                        lambda **kw: _response(content="We will mail jan@example.com."))
    out = agent.handle_turn("s", "hi")
    assert out["answer"] == "We will mail jan@example.com."  # the customer still sees it
    line = json.loads((fake_model / "turns.jsonl").read_text().splitlines()[-1])
    assert line["answer"] == "We will mail [EMAIL]."
    assert "[" not in line["timestamp"]


def test_model_failure_returns_friendly_fallback(fake_model, monkeypatch):
    def broken(**kw):
        raise TimeoutError("model API down")
    monkeypatch.setattr(agent.litellm, "completion", broken)
    out = agent.handle_turn("s", "What's the battery on the Ridge E 2025?")
    assert out["answer"] == agent.guardrails.MODEL_ERROR_FALLBACK
    assert "model_error:TimeoutError" in out["flags"]
    # history stays valid: user message + fallback answer, no dangling tool calls
    roles = [m["role"] for m in session.get_session("s").messages]
    assert roles == ["user", "assistant"]


def test_litellm_tool_path_imports():
    # litellm imports the undeclared 'mcp' package whenever tools are passed; keep it in requirements.txt
    import litellm.responses.mcp.chat_completions_handler  # noqa: F401


def test_second_return_request_after_execution_does_not_ask_again(fake_model):
    agent.handle_turn("s", "I want to return order CN-1042")
    agent.handle_turn("s", "yes")
    out = agent.handle_turn("s", "I want to return CN-1042")
    assert out["pending_confirmation"] is False
    assert len(tools.RETURNS) == 1


def test_executed_return_is_recorded_in_history_for_the_model(fake_model):
    agent.handle_turn("s", "I want to return order CN-1042")
    agent.handle_turn("s", "yes")
    notes = [m["content"] for m in session.get_session("s").messages if m["role"] == "system"]
    assert any("return_started" in n and "RT-0001" in n for n in notes)


def test_cancelled_return_is_recorded_in_history(fake_model):
    agent.handle_turn("s", "I want to return order CN-1042")
    agent.handle_turn("s", "no")
    notes = [m["content"] for m in session.get_session("s").messages if m["role"] == "system"]
    assert any("return_cancelled_by_customer" in n for n in notes)
