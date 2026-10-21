"""API tests with the agent stubbed out (no LLM calls)."""
import pytest
from fastapi.testclient import TestClient

from agent import config, session
from api import main


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path)
    session.SESSIONS.clear()

    def fake_turn(session_id, message):
        s = session.get_session(session_id)
        s.turn_count += 1
        return {"session_id": session_id, "turn_id": s.turn_count, "answer": "hi", "tools_called": [],
                "flags": [], "pending_confirmation": False, "latency_ms": 1}

    monkeypatch.setattr(main, "handle_turn", fake_turn)
    return TestClient(main.app)


def test_chat_then_feedback(client, tmp_path):
    turn = client.post("/chat", json={"session_id": "a", "message": "hello"}).json()
    assert turn["turn_id"] == 1
    r = client.post("/feedback", json={"session_id": "a", "turn_id": 1, "rating": "up"})
    assert r.status_code == 200
    assert '"rating": "up"' in (tmp_path / "feedback.jsonl").read_text()


def test_feedback_unknown_turn(client):
    assert client.post("/feedback", json={"session_id": "nope", "turn_id": 1, "rating": "up"}).status_code == 404


def test_feedback_rejects_bad_rating(client):
    client.post("/chat", json={"session_id": "a", "message": "hello"})
    assert client.post("/feedback", json={"session_id": "a", "turn_id": 1, "rating": "meh"}).status_code == 422


def test_empty_message_rejected(client):
    assert client.post("/chat", json={"session_id": "a", "message": ""}).status_code == 422
