"""What happened during one customer message: model calls, lookups and flags.

These are per-turn facts for the log line, not conversation state, so they stay out of the checkpointed graph
state. LangGraph runs nodes with a copy of the context, and the copy still points at this same object.
"""
from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass
class Turn:
    prompt_version: str = "graph_v1"
    model_calls: list[dict] = field(default_factory=list)
    tools_called: list[dict] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


_current: ContextVar[Turn | None] = ContextVar("turn", default=None)


def begin(prompt_version: str) -> Turn:
    turn = Turn(prompt_version=prompt_version)
    _current.set(turn)
    return turn


def current() -> Turn:
    """The running turn, or a throwaway one when a node is called directly."""
    return _current.get() or Turn()
