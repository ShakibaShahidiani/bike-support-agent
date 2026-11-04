"""What the API and eval need to know per session_id (demo only: in memory).

The conversation itself lives in the LangGraph checkpointer. `pending_action` and `executed_returns` are copied from
the graph state after each turn.
"""
from dataclasses import dataclass, field


@dataclass
class Session:
    session_id: str
    turn_count: int = 0
    pending_action: dict | None = None
    executed_returns: list = field(default_factory=list)
    trace_ids: dict = field(default_factory=dict)  # turn_id -> Langfuse trace id, to attach feedback


SESSIONS: dict[str, Session] = {}


def get_session(session_id: str) -> Session:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = Session(session_id=session_id)
    return SESSIONS[session_id]
