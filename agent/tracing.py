"""Langfuse tracing, through its LangChain callback: nodes, generations, tools and the retriever are captured
automatically. Does nothing when the keys aren't set, so tests and CI need no account.
"""
import json
import os
from contextlib import contextmanager

from langchain_core.documents import Document
from langfuse import Langfuse, get_client, propagate_attributes
from langfuse.langchain import CallbackHandler
from langgraph.types import Command

from agent.guardrails import redact_pii


def enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY")) and os.getenv("LANGFUSE_TRACING_ENABLED", "true").lower() != "false"


def _readable(data):
    """Turn framework objects into what a person wants to see in the Langfuse UI."""
    if isinstance(data, Command) and data.resume is not None:  # the graph continuing after the pause
        return {"resume_with": data.resume}
    if isinstance(data, Command):  # a node's decision: where next, and which values changed
        changes = {k: v for k, v in (data.update or {}).items() if v is not None and k != "history"}
        return {"next": data.goto, "changes": changes}
    if isinstance(data, Document):
        return {"source": data.metadata.get("source"), "content": data.page_content}
    if isinstance(data, dict):
        if "page_content" in data:  # a retrieved chunk, already serialized by the callback
            return {"source": data.get("metadata", {}).get("source"), "content": data["page_content"]}
        if "history" in data:  # graph state: the conversation is already visible in each generation's prompt
            return {k: v for k, v in data.items() if v is not None and k != "history"}
        if data.get("role") == "assistant" and "additional_kwargs" in data:  # a model reply: show the parsed JSON
            try:
                return json.loads(data["content"])
            except (TypeError, ValueError):
                return data["content"]
    return data


def _mask(data, **kwargs):
    """Applied by Langfuse to everything it sends: make it readable, and redact PII as in the JSONL logs.
    Anything that still isn't plain data becomes redacted text, so no field can get past the redaction."""
    data = _readable(data)
    if data is None or isinstance(data, (bool, int, float)):
        return data
    if isinstance(data, dict):
        return {k: _mask(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_mask(v) for v in data]
    return redact_pii(data if isinstance(data, str) else str(data))


_client: Langfuse | None = None


def _get_client() -> Langfuse:
    global _client
    if _client is None:
        _client = Langfuse(mask=_mask)
    return _client


class TurnTrace:
    """What a turn needs from tracing. With tracing off it holds nothing and does nothing."""

    def __init__(self, span=None, handler=None):
        self.span, self.handler = span, handler
        self.trace_id = span.trace_id if span else None
        self.callbacks = [handler] if handler else []

    def finish(self, answer: str, flags: list[str]) -> None:
        if self.span:
            self.span.update(output=answer, metadata={"flags": flags, "paused_for_confirmation": "pending_return" in flags})


@contextmanager
def trace_turn(session_id: str, turn_id: int, prompt_version: str, message: str):
    """One trace per customer message, with the message and answer as its input and output (not the whole graph
    state). The session groups a conversation's turns; the callback nests the graph's nodes under this span."""
    if not enabled():
        yield TurnTrace()
        return
    with _get_client().start_as_current_observation(as_type="span", name="support-turn", input=message,
                                                    metadata={"turn_id": turn_id}) as span:
        with propagate_attributes(trace_name="support-turn", session_id=session_id, tags=[prompt_version]):
            yield TurnTrace(span, CallbackHandler())


def score_feedback(trace_id: str | None, rating: str) -> None:
    """Attach a thumbs up/down to the trace of the answer it rates."""
    if trace_id and enabled():
        _get_client().create_score(name="user_feedback", value=1 if rating == "up" else 0,
                                   trace_id=trace_id, data_type="BOOLEAN")


def flush() -> None:
    if enabled():
        get_client().flush()
