"""The agent loop: one call to handle_turn() per user message.

Order of checks per turn:
  1. input guardrail (length)
  2. pending return? -> explicit confirmation executes it in code, anything else cancels it
  3. safety escalation -> model without tools + forced handover, post-checked
  4. normal tool loop (max 5 iterations)
  5. grounding post-check on numbers in the answer
"""
import json
import time

import litellm
from langchain_core.utils.function_calling import convert_to_openai_tool

from agent import config, guardrails
from agent.logger import log_turn
from agent.session import Session, get_session
from agent.tools import TOOLS, TOOLS_BY_NAME, execute_pending_return

TOOL_SCHEMAS = [convert_to_openai_tool(t) for t in TOOLS]


def load_system_prompt(version: str) -> str:
    return (config.PROMPTS_DIR / f"system_{version}.md").read_text()


class Turn:
    """Collects everything that happens in one turn, for the answer and the log line."""

    def __init__(self, session: Session, prompt_version: str):
        self.session = session
        self.prompt_version = prompt_version
        self.system_prompt = load_system_prompt(prompt_version)
        self.model_calls: list[dict] = []
        self.tools_called: list[dict] = []
        self.tool_results_text = ""
        self.flags: list[str] = []

    def call_model(self, extra_system: str | None = None, use_tools: bool = True):
        messages = [{"role": "system", "content": self.system_prompt}]
        if extra_system:
            messages.append({"role": "system", "content": extra_system})
        messages += self.session.messages
        start = time.perf_counter()
        response = litellm.completion(
            model=config.MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS if use_tools else None,
            timeout=config.MODEL_TIMEOUT_S,
            num_retries=2,
        )
        usage = response.usage
        self.model_calls.append({
            "latency_ms": round((time.perf_counter() - start) * 1000),
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "cost_usd": _cost(response, usage),
        })
        return response.choices[0].message

    def run_tool(self, name: str, args: dict) -> dict:
        self.tools_called.append({"name": name, "args": args})
        tool = TOOLS_BY_NAME.get(name)
        if tool is None:
            result = {"error": f"unknown tool {name}"}
        else:
            if name == "start_return":
                args = {**args, "session": self.session}  # injected, hidden from the model
            try:
                result = tool.invoke(args)
            except Exception as e:  # bad arguments from the model -> tell the model, don't crash
                result = {"error": "invalid_arguments", "detail": str(e)[:200]}
        self.tool_results_text += json.dumps(result, default=str) + "\n"
        return result


def _cost(response, usage) -> float | None:
    try:
        return litellm.completion_cost(completion_response=response)
    except Exception:
        price = config.FALLBACK_PRICES.get(config.MODEL.split("/")[-1])
        if not price:
            return None
        return usage.prompt_tokens * price["input"] + usage.completion_tokens * price["output"]


def _tool_loop(turn: Turn, extra_system: str | None = None) -> str:
    session = turn.session
    for _ in range(config.MAX_TOOL_ITERATIONS):
        msg = turn.call_model(extra_system)
        if not msg.tool_calls:
            return msg.content or ""
        session.messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = turn.run_tool(tc.function.name, args)
            session.messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result, default=str)})
    turn.flags.append("max_iterations")
    return turn.call_model(extra_system, use_tools=False).content or ""


def handle_turn(session_id: str, message: str, prompt_version: str | None = None) -> dict:
    start = time.perf_counter()
    session = get_session(session_id)
    session.turn_count += 1
    turn = Turn(session, prompt_version or config.PROMPT_VERSION)

    if guardrails.input_too_long(message):
        turn.flags.append("input_too_long")
        answer = f"Your message is too long. Please keep it under {config.MAX_INPUT_CHARS} characters."
        return _finish(turn, message, answer, start, store_history=False)

    session.messages.append({"role": "user", "content": message})
    extra_system = None

    # 2. Confirmation gate: decided in code, never by the model.
    if session.pending_action:
        if guardrails.is_confirmation(message):
            result = execute_pending_return(session)
            turn.tools_called.append({"name": "execute_pending_return", "args": {}})
            turn.tool_results_text += json.dumps(result) + "\n"
            turn.flags.append("return_executed" if result.get("status") == "return_started" else "return_failed")
            extra_system = (f"The customer confirmed. The system ran the return action with this result: "
                            f"{json.dumps(result)}. Tell the customer the outcome. Do not call start_return again.")
            answer = turn.call_model(extra_system, use_tools=False).content or ""
            return _finish(turn, message, _grounding_check(turn, answer), start)
        session.pending_action = None
        turn.flags.append("pending_return_cancelled")
        extra_system = "The customer did not confirm, so the pending return was cancelled. Mention this briefly."

    # 3. Safety escalation.
    if guardrails.needs_safety_escalation(message):
        turn.flags.append("safety_escalation")
        answer = turn.call_model(guardrails.ESCALATION_INSTRUCTION, use_tools=False).content or ""
        if not guardrails.escalation_reply_ok(answer):
            turn.flags.append("escalation_fallback")
            answer = guardrails.ESCALATION_FALLBACK
        return _finish(turn, message, answer, start)

    # 4. Normal tool loop, 5. grounding check.
    answer = _tool_loop(turn, extra_system)
    return _finish(turn, message, _grounding_check(turn, answer), start)


def _grounding_check(turn: Turn, answer: str) -> str:
    missing = guardrails.ungrounded_numbers(answer, turn.tool_results_text)
    if missing:
        turn.flags.append(f"ungrounded:{','.join(missing)}")
        return guardrails.GROUNDING_FALLBACK
    return answer


def _finish(turn: Turn, message: str, answer: str, start: float, store_history: bool = True) -> dict:
    session = turn.session
    if store_history:
        session.messages.append({"role": "assistant", "content": answer})
    calls = turn.model_calls
    costs = [c["cost_usd"] for c in calls if c["cost_usd"] is not None]
    record = {
        "session_id": session.session_id,
        "turn_id": session.turn_count,
        "prompt_version": turn.prompt_version,
        "model": config.MODEL,
        "user_message": message,
        "tools_called": turn.tools_called,
        "latency_ms": {"total": round((time.perf_counter() - start) * 1000),
                       "model_calls": [c["latency_ms"] for c in calls]},
        "input_tokens": sum(c["input_tokens"] for c in calls),
        "output_tokens": sum(c["output_tokens"] for c in calls),
        "cost_usd": round(sum(costs), 6) if costs else None,
        "answer_length": len(answer),
        "flags": turn.flags,
        "pending_action": session.pending_action,
    }
    log_turn(record)
    return {"session_id": session.session_id, "turn_id": session.turn_count, "answer": answer,
            "tools_called": turn.tools_called, "flags": turn.flags,
            "pending_confirmation": session.pending_action is not None,
            "latency_ms": record["latency_ms"]["total"]}


if __name__ == "__main__":
    # Tiny REPL for manual testing: python -m agent.agent
    import uuid

    sid = str(uuid.uuid4())[:8]
    print(f"session {sid}, model {config.MODEL}, prompt {config.PROMPT_VERSION}. Ctrl-C to quit.")
    while True:
        out = handle_turn(sid, input("\nyou> "))
        print(f"agent> {out['answer']}")
        print(f"  [tools={[t['name'] for t in out['tools_called']]} flags={out['flags']} {out['latency_ms']}ms]")
