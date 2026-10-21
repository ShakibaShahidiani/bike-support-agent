"""Agent tools. Each is a LangChain @tool, so its JSON schema comes from the signature + docstring.

Every tool returns a plain dict. Errors are returned (not raised) so the model can read them
and ask the user or abstain.
"""
import json
import re
from datetime import date
from typing import Annotated

from langchain_core.tools import InjectedToolArg, tool

from agent import config, rag
from agent.session import Session

BIKES = json.loads((config.DATA_DIR / "bikes.json").read_text())
ORDERS = {o["order_id"]: o for o in json.loads((config.DATA_DIR / "orders.json").read_text())}
KNOWN_MODELS = sorted({b["model"] for b in BIKES})
KNOWN_YEARS = sorted({b["year"] for b in BIKES})
ORDER_ID_RE = re.compile(r"^CN-\d{4}$")

# order_id -> return_id. In-memory idempotency store (production: DB table with a unique key).
RETURNS: dict[str, str] = {}


# ---------- validation helpers ----------

def _resolve_model(model: str) -> str | None:
    """Case-insensitive exact match against the catalogue. No fuzzy matching on purpose."""
    for known in KNOWN_MODELS:
        if known.lower() == model.strip().lower():
            return known
    return None


def _validate_bike(model: str, year: int) -> tuple[dict | None, dict | None]:
    """Returns (spec, error)."""
    name = _resolve_model(model)
    if name is None:
        return None, {"error": "unknown_model", "model": model, "known_models": KNOWN_MODELS}
    if year not in KNOWN_YEARS:
        return None, {"error": "unknown_year", "year": year, "known_years": KNOWN_YEARS}
    spec = next(b for b in BIKES if b["model"] == name and b["year"] == year)
    return spec, None


def _validate_order(order_id: str) -> tuple[dict | None, dict | None]:
    order_id = order_id.strip().upper()
    if not ORDER_ID_RE.match(order_id):
        return None, {"error": "invalid_order_id", "expected_format": "CN-1234"}
    if order_id not in ORDERS:
        return None, {"error": "order_not_found", "order_id": order_id}
    return ORDERS[order_id], None


# ---------- read-only tools ----------

@tool
def get_bike_spec(model: str, year: int) -> dict:
    """Get the technical spec of a Ferro Cycles bike for one model year."""
    spec, error = _validate_bike(model, year)
    return error or spec


@tool
def check_compatibility(part_type: str, part_value: str, model: str, year: int) -> dict:
    """Check if a part fits a bike. part_type is one of: front_rotor, rear_rotor, tyre.
    part_value is the size in mm, e.g. '180' or '180mm'."""
    spec, error = _validate_bike(model, year)
    if error:
        return error
    limits = {"front_rotor": "max_rotor_front_mm", "rear_rotor": "max_rotor_rear_mm", "tyre": "tyre_clearance_mm"}
    if part_type not in limits:
        return {"error": "unknown_part_type", "known_part_types": list(limits)}
    match = re.search(r"\d+", part_value)
    if not match:
        return {"error": "invalid_part_value", "expected": "a size in mm, e.g. 180"}
    size = int(match.group())
    limit = spec[limits[part_type]]
    if limit is None:
        return {"compatible": False, "reason": f"{spec['model']} {year} has {spec['brake_type']} brakes, no disc rotors."}
    ok = size <= limit
    return {
        "compatible": ok,
        "reason": f"{part_type} {size}mm {'is within' if ok else 'exceeds'} the {limit}mm maximum "
                  f"for {spec['model']} {year}.",
    }


@tool
def get_order_status(order_id: str) -> dict:
    """Get the status and dates of an order (id format CN-1234)."""
    order, error = _validate_order(order_id)
    if error:
        return error
    return {k: order[k] for k in ("order_id", "status", "order_date", "delivery_date")}


@tool
def search_support_docs(query: str, model: str | None = None, year: int | None = None) -> dict:
    """Search current support docs (returns policy, warranty, sizing, care).
    Optionally filter by bike model and year. Superseded policies are never returned."""
    if model and _resolve_model(model) is None:
        return {"error": "unknown_model", "model": model, "known_models": KNOWN_MODELS}
    results = rag.search(query, _resolve_model(model) if model else None, year)
    return {"results": results} if results else {"results": [], "note": "no matching documents"}


# ---------- action tool (confirmation gate) ----------

def check_return_eligibility(order_id: str, today: date | None = None) -> dict:
    order, error = _validate_order(order_id)
    if error:
        return error
    if order["status"] != "delivered":
        return {"eligible": False, "reason": f"order status is '{order['status']}', only delivered orders can be returned"}
    days = ((today or date.today()) - date.fromisoformat(order["delivery_date"])).days
    if days > config.RETURN_WINDOW_DAYS:
        return {"eligible": False, "reason": f"delivered {days} days ago, outside the {config.RETURN_WINDOW_DAYS}-day return window"}
    return {"eligible": True, "order_id": order["order_id"]}


@tool
def start_return(order_id: str, reason: str, session: Annotated[Session, InjectedToolArg]) -> dict:
    """Request a return for a delivered order. This does NOT execute the return: it asks the
    customer to confirm first. Tell the customer to reply 'yes' to confirm."""
    check = check_return_eligibility(order_id)
    if not check.get("eligible"):
        return check
    session.pending_action = {"type": "start_return", "order_id": check["order_id"], "reason": reason}
    return {
        "status": "confirmation_required",
        "order_id": check["order_id"],
        "instruction": "Ask the customer to reply 'yes' to confirm the return, or anything else to cancel.",
    }


def execute_pending_return(session: Session, today: date | None = None) -> dict:
    """Called by the agent loop (never by the model) after the user confirmed."""
    action = session.pending_action
    if not action or action["type"] != "start_return":
        return {"error": "no_pending_confirmation"}
    session.pending_action = None
    check = check_return_eligibility(action["order_id"], today)  # re-check: time may have passed
    if not check.get("eligible"):
        return check
    order_id = action["order_id"]
    if order_id not in RETURNS:
        RETURNS[order_id] = f"RT-{len(RETURNS) + 1:04d}"
    session.executed_returns.append(RETURNS[order_id])
    return {"status": "return_started", "order_id": order_id, "return_id": RETURNS[order_id]}


TOOLS = [get_bike_spec, check_compatibility, get_order_status, search_support_docs, start_return]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}
