"""In-memory session state per session_id (demo only; production would use Redis or a database)."""
from dataclasses import dataclass, field


@dataclass
class Session:
    session_id: str
    messages: list = field(default_factory=list)  # chat history in OpenAI message format
    turn_count: int = 0
    pending_action: dict | None = None  # {"type": "start_return", "order_id", "reason", "created_turn"}
    executed_returns: list = field(default_factory=list)  # return ids created in this session
    # Return-intent gate: start_return may only run when the user asked for a return this turn,
    # or is answering the agent's follow-up question (order id / reason) from the previous turn.
    return_allowed: bool = False
    awaiting_return_info: bool = False


SESSIONS: dict[str, Session] = {}


def get_session(session_id: str) -> Session:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = Session(session_id=session_id)
    return SESSIONS[session_id]
