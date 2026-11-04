"""The support workflow as a LangGraph graph. The model labels and extracts; code decides and acts.

    intake -> fixed_reply | docs | order_status | parts_collect -> parts_answer
                                               | return_check -> confirm (pause) -> execute_return

Conversation state is checkpointed per session, so a turn can end while slots stay filled (a question was asked),
and `confirm` can pause the graph until the customer's next message.
"""
import functools
import re
from operator import add
from typing import Annotated, ClassVar, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from agent import config, guardrails, llm, tools, turn

Task = Literal["docs", "parts", "order", "return"]
TASK_NODES = {"docs": "docs", "parts": "parts_collect", "order": "order_status", "return": "return_check"}
SLOTS = ("model", "year", "other_year", "attribute", "part_value", "order_id", "reason")
NO_SLOTS = {slot: None for slot in SLOTS}
HISTORY_TURNS = 8

HUMAN_HANDOVER = (
    "Our human support team can take this from here: please email support@ferrocycles.example with your order ID, "
    "if you have one. If anyone is hurt, or the bike or battery may be unsafe, please stop using it until you've "
    "spoken to them."
)
OFF_TOPIC = (
    "Sorry, I can't help with that. I can only help with Ferro Cycles bike specs, part compatibility, orders, returns "
    "and our support documents. For anything else about your bike or order, a human support agent can help: "
    "support@ferrocycles.example."
)
DONT_KNOW = "I couldn't find that in our support documents. Would you like the details for a human support agent?"
OFFER_HUMAN = "Would you like the details for a human support agent?"


class State(TypedDict, total=False):
    message: str
    history: Annotated[list[dict], add]
    answer: str | None  # cleared when a turn starts
    notice: str | None  # said before the next answer, e.g. that a return was cancelled
    reply_kind: str
    active_task: Task | None
    model: str | None
    year: int | None
    other_year: int | None
    attribute: str | None
    part_value: str | None
    order_id: str | None
    reason: str | None
    executed_returns: list[str]


# ---------- model outputs ----------

class Route(BaseModel):
    step: ClassVar[str] = "classify-intent"
    intent: Literal["continue", "docs", "parts", "order", "return", "human", "other"]


# No field defaults here: OpenAI's strict JSON schema requires every field, so "not said" is an explicit null.
class PartsSlots(BaseModel):
    step: ClassVar[str] = "extract-bike-details"
    model: str | None
    year: int | None
    other_year: int | None
    attribute: Literal["brake_type", "frame", "fork", "max_rotor_front_mm", "max_rotor_rear_mm",
                       "tyre_clearance_mm", "battery_wh", "overview"] | None
    part_value: str | None


class ReturnSlots(BaseModel):
    step: ClassVar[str] = "extract-return-details"
    order_id: str | None
    reason: str | None


class ConfirmLabel(BaseModel):
    step: ClassVar[str] = "classify-confirmation"
    label: Literal["confirm", "cancel", "other"]


class Query(BaseModel):
    step: ClassVar[str] = "rewrite-question"
    query: str


class DocAnswer(BaseModel):
    step: ClassVar[str] = "answer-from-documents"
    answer: str | None  # cleared when a turn starts
    used_chunks: list[int]


# ---------- helpers ----------

def _prompt(name: str) -> str:
    return (config.PROMPTS_DIR / turn.current().prompt_version / f"{name}.md").read_text()


def _conversation(state: State, message: str | None = None) -> str:
    """Recent conversation as text. `message` is added when it isn't in the history yet."""
    entries = state.get("history", [])[-HISTORY_TURNS:]
    lines = [f"{'customer' if m['role'] == 'user' else 'assistant'}: {m['content']}" for m in entries]
    if message:
        lines.append(f"customer: {message}")
    return "\n".join(lines)


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant(text: str) -> dict:
    return {"role": "assistant", "content": text}


def _reply(state: State, text: str, **updates) -> dict:
    """The state update that ends a turn with `text`, after any waiting notice."""
    if state.get("notice"):
        text = f"{state['notice']} {text}"
    return {"answer": text, "notice": None, "history": [_assistant(text)], **updates}


def _lookup(tool, args: dict, log_as: str | None = None) -> dict:
    """Run a catalogue tool and record it. Calling it as a LangChain tool also makes it a span in the Langfuse trace."""
    turn.current().tools_called.append({"name": log_as or tool.name, "args": args})
    return tool.invoke(args)


def _model_failed(state: State) -> dict:
    turn.current().flags.append("model_error")
    return _reply(state, guardrails.MODEL_ERROR_FALLBACK)


def _safe_model_node(node):
    """A failed model call becomes a friendly reply, so the graph never stops halfway through a node."""
    @functools.wraps(node)
    def wrapper(state: State):
        try:
            return node(state)
        except GraphBubbleUp:  # how LangGraph signals an interrupt; not a failure
            raise
        except Exception as e:
            turn.current().flags.append(f"error:{type(e).__name__}")
            return Command(goto=END, update=_model_failed(state))
    return wrapper


# ---------- nodes ----------

def intake(state: State) -> Command[Literal["fixed_reply", "docs", "parts_collect", "order_status", "return_check", "__end__"]]:
    """The single entry point: length check, then one model call for the intent."""
    message = state["message"]
    if guardrails.input_too_long(message):
        turn.current().flags.append("input_too_long")
        return Command(goto="fixed_reply", update={"reply_kind": "too_long"})

    active = state.get("active_task")
    update = {"history": [_user(message)]}
    try:
        route = llm.structured(
            Route, _prompt("router"),
            f"Task in progress: {active or 'none'}\n\nConversation:\n{_conversation(state, message)}")
    except Exception as e:
        turn.current().flags.append(f"error:{type(e).__name__}")
        failed = _model_failed(state)
        return Command(goto=END, update={**failed, "history": update["history"] + failed["history"]})

    intent = route.intent
    turn.current().flags.append(f"intent:{intent}")
    task = active if intent == "continue" else intent
    if task in (None, "other", "human"):
        kind = "human" if task == "human" else "other"
        return Command(goto="fixed_reply", update={**update, **NO_SLOTS, "active_task": None, "reply_kind": kind})
    if task != active:
        update.update(NO_SLOTS)  # a new task must not inherit the old task's details
    return Command(goto=TASK_NODES[task], update={**update, "active_task": task})


def fixed_reply(state: State) -> dict:
    kind = state["reply_kind"]
    if kind == "too_long":
        text = f"Your message is too long. Please keep it under {config.MAX_INPUT_CHARS} characters."
        return {**_reply(state, text), "history": []}  # too long to keep in the conversation
    turn.current().flags.append("human_handover" if kind == "human" else "refused")
    return _reply(state, HUMAN_HANDOVER if kind == "human" else OFF_TOPIC)


@_safe_model_node
def docs(state: State) -> dict:
    """Standalone question -> currently valid chunks -> answer with verified citations, or "I don't know"."""
    query = state["message"]
    if len(state.get("history", [])) > 1:
        query = llm.structured(Query, _prompt("rewrite"), _conversation(state)).query

    chunks = _lookup(tools.search_support_docs, {"query": query})["results"]
    if not chunks:
        turn.current().flags.append("no_documents")
        return _reply(state, DONT_KNOW)

    numbered = "\n\n".join(f"[{i}] ({c['source']})\n{c['content']}" for i, c in enumerate(chunks, 1))
    out = llm.structured(DocAnswer, _prompt("docs"), f"Question: {query}\n\nChunks:\n{numbered}")
    used = [chunks[i - 1] for i in dict.fromkeys(out.used_chunks) if 1 <= i <= len(chunks)]
    if not used or not out.answer.strip():
        turn.current().flags.append("no_valid_citation")
        return _reply(state, DONT_KNOW)

    source_text = " ".join(c["content"] for c in chunks)
    if missing := guardrails.ungrounded_numbers(out.answer, source_text):
        turn.current().flags.append(f"ungrounded:{','.join(missing)}")
        return _reply(state, guardrails.GROUNDING_FALLBACK)
    sources = ", ".join(dict.fromkeys(c["source"] for c in used))
    return _reply(state, f"{out.answer.strip()}\n\nSources: {sources}")


@_safe_model_node
def parts_collect(state: State) -> Command[Literal["parts_answer", "__end__"]]:
    """Fill the slots from the conversation. If one is missing or wrong, ask for that one and end the turn."""
    current = {k: state.get(k) for k in ("model", "year", "other_year", "attribute", "part_value")}
    out = llm.structured(PartsSlots, _prompt("parts"),
                         f"Current values: {current}\n\nConversation:\n{_conversation(state)}")
    slots = {k: getattr(out, k) if getattr(out, k) is not None else current[k] for k in current}
    models, years = ", ".join(tools.KNOWN_MODELS), " or ".join(str(y) for y in tools.KNOWN_YEARS)

    def ask(text: str, **changes) -> Command:
        return Command(goto=END, update=_reply(state, text, **{**slots, **changes}))

    if not slots["model"]:
        return ask(f"Which Ferro Cycles model is it? We have: {models}.")
    known_model = tools.resolve_model(slots["model"])
    if known_model is None:
        turn.current().flags.append("unknown_model")
        return ask(f"I couldn't find a bike called {slots['model']} in our catalogue. We have: {models}. "
                   f"Which one do you mean? {OFFER_HUMAN}", model=None)
    slots["model"] = known_model
    if not slots["year"]:
        return ask(f"Which model year is your {known_model}: {years}?")
    if slots["year"] not in tools.KNOWN_YEARS:
        turn.current().flags.append("unknown_year")
        return ask(f"I couldn't find a {slots['year']} {known_model} in our catalogue. It's listed for {years}. "
                   f"Which one do you mean? {OFFER_HUMAN}", year=None)
    if not slots["attribute"]:
        return ask(f"What would you like to know about the {slots['year']} {known_model}?")
    return Command(goto="parts_answer", update=slots)


SPEC_LABELS = {
    "brake_type": ("brake type", ""), "frame": ("frame", ""), "fork": ("fork", ""),
    "max_rotor_front_mm": ("maximum front rotor size", "mm"), "max_rotor_rear_mm": ("maximum rear rotor size", "mm"),
    "tyre_clearance_mm": ("tyre clearance", "mm"), "battery_wh": ("battery", "Wh"),
}
PART_TYPES = {"max_rotor_front_mm": "front_rotor", "max_rotor_rear_mm": "rear_rotor", "tyre_clearance_mm": "tyre"}


def _spec_sentence(spec: dict, attribute: str) -> str:
    label, unit = SPEC_LABELS[attribute]
    value = spec[attribute]
    if value is None:
        return f"The {spec['year']} {spec['model']} has no {label.replace('maximum ', '').replace(' size', '')}."
    return f"The {label} on the {spec['year']} {spec['model']} is {value}{unit}."


def parts_answer(state: State) -> dict:
    """Runs only with valid slots. Numbers come straight from the catalogue, so none can be invented."""
    model, year, attribute = state["model"], state["year"], state["attribute"]
    if state.get("part_value") and attribute in PART_TYPES:
        result = _lookup(tools.check_compatibility, {"part_type": PART_TYPES[attribute], "part_value": state["part_value"],
                                                     "model": model, "year": year})
        if "error" in result:
            text = f"I couldn't check that part. {OFFER_HUMAN}"
        else:
            text = f"{'Yes, that fits' if result['compatible'] else 'No, that will not fit'}: {result['reason']}"
    else:
        spec = _lookup(tools.get_bike_spec, {"model": model, "year": year})
        if attribute == "overview":
            text = " ".join(_spec_sentence(spec, a) for a in SPEC_LABELS)
        else:
            text = _spec_sentence(spec, attribute)
    other = state.get("other_year")
    if other in tools.KNOWN_YEARS and other != year:
        text += f" I can also give you the figure for the {other} model, just let me know."
    return _reply(state, text, attribute=None, part_value=None, other_year=None)


ORDER_ID_IN_TEXT_RE = re.compile(r"\bCN-?\s?(\d{4})\b", re.I)


def _order_id_in(text: str) -> str | None:
    match = ORDER_ID_IN_TEXT_RE.search(text)
    return f"CN-{match.group(1)}" if match else None


def order_status(state: State) -> dict:
    order_id = _order_id_in(state["message"]) or state.get("order_id")
    if not order_id:
        return _reply(state, "What's your order ID? It looks like CN-1234.")
    result = _lookup(tools.get_order_status, {"order_id": order_id})
    if "error" in result:
        return _reply(state, f"I couldn't find order {order_id}. Please check the ID. {OFFER_HUMAN}", order_id=None)
    status = result["status"]
    if status == "delivered":
        return _reply(state, f"Order {order_id} was delivered on {result['delivery_date']}.", order_id=order_id)
    state_text = {"cancelled": "has been cancelled"}.get(status, f"is currently {status}")
    return _reply(state, f"Order {order_id} {state_text}. I can't confirm delivery dates or refunds, but a human support "
                         "agent can help with that.", order_id=order_id)


@_safe_model_node
def return_check(state: State) -> Command[Literal["confirm", "__end__"]]:
    """Find the order, check it in code, get a reason. Nothing is changed here."""
    out = llm.structured(ReturnSlots, _prompt("return"), _conversation(state))
    order_id = _order_id_in(state["message"]) or out.order_id or state.get("order_id")
    reason = out.reason or state.get("reason")
    if not order_id:
        return Command(goto=END, update=_reply(state, "Which order would you like to return? The ID looks like CN-1234.",
                                               reason=reason))
    check = _lookup(tools.check_return_eligibility, {"order_id": order_id}, log_as="start_return")
    if check.get("error"):
        text = f"I couldn't find order {order_id}. Please check the ID. {OFFER_HUMAN}"
        return Command(goto=END, update=_reply(state, text, order_id=None, reason=reason))
    if check.get("status") == "already_returned":
        text = f"A return for order {check['order_id']} already exists. Your return ID is {check['return_id']}."
        return Command(goto=END, update=_reply(state, text, **NO_SLOTS, active_task=None))
    if not check["eligible"]:
        text = f"I can't start a return for order {order_id}: {check['reason']}. {OFFER_HUMAN}"
        return Command(goto=END, update=_reply(state, text, **NO_SLOTS, active_task=None))
    if not reason:
        return Command(goto=END, update=_reply(state, "That order can be returned. What's your reason for the return?",
                                               order_id=check["order_id"]))
    turn.current().flags.append("pending_return")
    return Command(goto="confirm", update={"order_id": check["order_id"], "reason": reason})


def confirm_prompt(state: State) -> str:
    text = (f"Please confirm: start a return for order {state['order_id']} (reason: {state['reason']})? "
            "Reply yes to confirm or no to cancel.")
    return f"{state['notice']} {text}" if state.get("notice") else text


def confirm(state: State) -> Command[Literal["execute_return", "intake", "__end__"]]:
    """Pause until the customer's next message. Only a clear yes reaches execute_return.

    LangGraph re-runs this node from the top on resume, so nothing before interrupt() may have side effects.
    """
    prompt = confirm_prompt(state)
    message = interrupt(prompt)

    order_id = state["order_id"]
    asked = {"history": [_assistant(prompt)], "notice": None, "answer": None}
    cleared = {**NO_SLOTS, "active_task": None}
    label = "other"  # also covers messages too long to send to the model; intake handles those
    if not guardrails.input_too_long(message):
        try:
            label = llm.structured(ConfirmLabel, _prompt("confirm"),
                                   f"Asked: {prompt}\nCustomer replied: {message}").label
        except Exception as e:  # fail closed: a return only runs on a clear yes
            turn.current().flags += [f"error:{type(e).__name__}", "pending_return_cancelled"]
            text = f"Sorry, I had technical trouble, so I did not start the return for order {order_id}. Please ask again."
            return Command(goto=END, update={"answer": text, "history": asked["history"] + [_user(message), _assistant(text)],
                                             "notice": None, **cleared})
    if label == "confirm":
        return Command(goto="execute_return",
                       update={**asked, "message": message, "history": asked["history"] + [_user(message)]})
    turn.current().flags.append("pending_return_cancelled")
    if label == "cancel":
        text = f"Okay, I've cancelled the return request for order {order_id}. Is there anything else I can help with?"
        return Command(goto=END, update={"answer": text, "history": asked["history"] + [_user(message), _assistant(text)],
                                         "notice": None, **cleared})
    notice = f"I did not start the return for order {order_id}. Just ask again if you still want it."
    return Command(goto="intake", update={**asked, **cleared, "message": message, "notice": notice})


def execute_return(state: State) -> dict:
    """The only node that changes anything. create_return re-checks eligibility and is idempotent."""
    result = _lookup(tools.create_return, {"order_id": state["order_id"], "reason": state["reason"]},
                     log_as="execute_pending_return")
    cleared = {**NO_SLOTS, "active_task": None}
    if result.get("status") != "return_started":
        turn.current().flags.append("return_failed")
        reason = result.get("reason") or result.get("error", "unknown error")
        return _reply(state, f"I couldn't start the return: {reason}. {OFFER_HUMAN}", **cleared)
    turn.current().flags.append("return_executed")
    text = (f"Your return for order {result['order_id']} has been started. Your return ID is {result['return_id']}. "
            "Our team will send you the next steps.")
    return _reply(state, text, **cleared, executed_returns=state.get("executed_returns", []) + [result["return_id"]])


# ---------- graph ----------

def build_graph(checkpointer=None):
    builder = StateGraph(State)
    for node in (intake, fixed_reply, docs, parts_collect, parts_answer, order_status, return_check, confirm, execute_return):
        builder.add_node(node.__name__, node)
    builder.add_edge(START, "intake")
    for last in ("fixed_reply", "docs", "parts_answer", "order_status", "execute_return"):
        builder.add_edge(last, END)
    # In memory, like the rest of this demo: a restart loses open conversations. Production would use Postgres.
    return builder.compile(checkpointer=checkpointer or InMemorySaver())
