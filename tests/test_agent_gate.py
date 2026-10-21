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


def test_anything_but_yes_cancels(fake_model):
    agent.handle_turn("s", "I want to return order CN-1042")
    out = agent.handle_turn("s", "hmm, what does shipping cost?")
    assert "pending_return_cancelled" in out["flags"]
    assert tools.RETURNS == {}


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
