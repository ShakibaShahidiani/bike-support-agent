"""In-memory session state per session_id (demo only; production would use Redis or a database)."""
from dataclasses import dataclass, field


@dataclass
class Session:
    session_id: str
    messages: list = field(default_factory=list)  # chat history in OpenAI message format
    turn_count: int = 0
    pending_action: dict | None = None  # e.g. {"type": "start_return", "order_id": ..., "reason": ...}
    executed_returns: list = field(default_factory=list)  # return ids created in this session


SESSIONS: dict[str, Session] = {}


def get_session(session_id: str) -> Session:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = Session(session_id=session_id)
    return SESSIONS[session_id]
