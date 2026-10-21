"""Append-only JSONL logs: one line per turn, one line per feedback rating."""
import json
from datetime import datetime, timezone

from agent import config
from agent.guardrails import redact_pii


def _append(filename: str, record: dict) -> None:
    config.LOGS_DIR.mkdir(exist_ok=True)
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
    line = redact_pii(json.dumps(record, ensure_ascii=False))
    with open(config.LOGS_DIR / filename, "a") as f:
        f.write(line + "\n")


def log_turn(record: dict) -> None:
    _append("turns.jsonl", record)


def log_feedback(session_id: str, turn_id: int, rating: str) -> None:
    _append("feedback.jsonl", {"session_id": session_id, "turn_id": turn_id, "rating": rating})
