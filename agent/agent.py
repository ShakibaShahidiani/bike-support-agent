"""Runs one customer message through the graph and logs the turn. The workflow itself is in agent/graph.py."""
import time

from langgraph.types import Command

from agent import config, guardrails, tracing, turn
from agent.graph import build_graph
from agent.logger import log_turn
from agent.session import get_session

graph = build_graph()


def reset() -> None:
    """Forget all conversations (tests and eval)."""
    global graph
    graph = build_graph()


def handle_turn(session_id: str, message: str, prompt_version: str | None = None) -> dict:
    start = time.perf_counter()
    session = get_session(session_id)
    session.turn_count += 1
    version = prompt_version or config.PROMPT_VERSION
    current = turn.begin(version)

    with tracing.trace_turn(session_id, session.turn_count, version, message) as trace:
        run_config = {"configurable": {"thread_id": session_id}, "callbacks": trace.callbacks, "run_name": "run-graph"}
        paused = bool(graph.get_state(run_config).next)  # waiting in `confirm`: this message resumes it
        try:
            result = graph.invoke(Command(resume=message) if paused else {"message": message, "answer": None}, run_config)
            interrupts = result.get("__interrupt__")
            answer = interrupts[0].value if interrupts else result["answer"]
        except Exception as e:  # a last resort; nodes handle model failures themselves
            current.flags.append(f"error:{type(e).__name__}")
            answer = guardrails.MODEL_ERROR_FALLBACK
        trace.finish(answer, current.flags)
    session.trace_ids[session.turn_count] = trace.trace_id

    snapshot = graph.get_state(run_config)
    session.executed_returns = snapshot.values.get("executed_returns", [])
    session.pending_action = (
        {"type": "start_return", "order_id": snapshot.values.get("order_id"), "reason": snapshot.values.get("reason")}
        if snapshot.next else None)

    return _finish(session, current, version, message, answer, start)


def _finish(session, current: turn.Turn, version: str, message: str, answer: str, start: float) -> dict:
    calls = current.model_calls
    costs = [c["cost_usd"] for c in calls if c["cost_usd"] is not None]
    latency = round((time.perf_counter() - start) * 1000)
    log_turn({
        "session_id": session.session_id,
        "turn_id": session.turn_count,
        "prompt_version": version,
        "model": config.MODEL,
        "user_message": message,
        "tools_called": current.tools_called,
        "latency_ms": {"total": latency, "model_calls": [c["latency_ms"] for c in calls]},
        "input_tokens": sum(c["input_tokens"] for c in calls),
        "output_tokens": sum(c["output_tokens"] for c in calls),
        "cost_usd": round(sum(costs), 6) if costs else None,
        "answer": answer,
        "answer_length": len(answer),
        "flags": current.flags,
        "pending_action": session.pending_action,
    })
    return {"session_id": session.session_id, "turn_id": session.turn_count, "answer": answer,
            "tools_called": current.tools_called, "flags": current.flags,
            "pending_confirmation": session.pending_action is not None, "latency_ms": latency}


if __name__ == "__main__":
    # Tiny REPL for manual testing: python -m agent.agent
    import uuid

    sid = str(uuid.uuid4())[:8]
    print(f"session {sid}, model {config.MODEL}, prompts {config.PROMPT_VERSION}. Ctrl-C to quit.")
    try:
        while True:
            out = handle_turn(sid, input("\nyou> "))
            print(f"agent> {out['answer']}")
            print(f"  [tools={[t['name'] for t in out['tools_called']]} flags={out['flags']} {out['latency_ms']}ms]")
    finally:
        tracing.flush()
