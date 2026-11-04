"""Catalogue, order and return operations, called by graph nodes (the model never chooses them).

They stay LangChain @tools because that validates their arguments and shows each call as a tool span in Langfuse.
Every tool returns a plain dict. Errors are returned, not raised, so the node can choose the reply.
"""
import json
import re
from datetime import date

from langchain_core.tools import tool

from agent import config, rag

BIKES = json.loads((config.DATA_DIR / "bikes.json").read_text())
ORDERS = {o["order_id"]: o for o in json.loads((config.DATA_DIR / "orders.json").read_text())}
KNOWN_MODELS = sorted({b["model"] for b in BIKES})
KNOWN_YEARS = sorted({b["year"] for b in BIKES})
ORDER_ID_RE = re.compile(r"^CN-\d{4}$")

# order_id -> return_id. In-memory idempotency store (production: DB table with a unique key).
RETURNS: dict[str, str] = {}


# ---------- validation helpers ----------

def resolve_model(model: str) -> str | None:
    """Case-insensitive exact match against the catalogue. No fuzzy matching on purpose."""
    for known in KNOWN_MODELS:
        if known.lower() == model.strip().lower():
            return known
    return None


def _validate_bike(model: str, year: int) -> tuple[dict | None, dict | None]:
    """Returns (spec, error)."""
    name = resolve_model(model)
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
    if model and resolve_model(model) is None:
        return {"error": "unknown_model", "model": model, "known_models": KNOWN_MODELS}
    results = rag.search(query, resolve_model(model) if model else None, year)
    return {"results": results} if results else {"results": [], "note": "no matching documents"}


# ---------- returns ----------

@tool
def check_return_eligibility(order_id: str, today: date | None = None) -> dict:
    """Check if an order can be returned: delivered, inside the return window, not already returned. Changes nothing."""
    order, error = _validate_order(order_id)
    if error:
        return error
    if order["order_id"] in RETURNS:
        return {"eligible": False, "status": "already_returned", "order_id": order["order_id"],
                "return_id": RETURNS[order["order_id"]], "reason": "a return already exists for this order"}
    if order["status"] != "delivered":
        return {"eligible": False, "reason": f"order status is '{order['status']}', only delivered orders can be returned"}
    days = ((today or date.today()) - date.fromisoformat(order["delivery_date"])).days
    if days > config.RETURN_WINDOW_DAYS:
        return {"eligible": False, "reason": f"delivered {days} days ago, outside the {config.RETURN_WINDOW_DAYS}-day return window"}
    return {"eligible": True, "order_id": order["order_id"]}


@tool
def create_return(order_id: str, reason: str, today: date | None = None) -> dict:
    """Create the return. Re-checks eligibility, because time may have passed since the customer was asked.
    Idempotent: a second call for the same order gives the same return id."""
    check = check_return_eligibility.invoke({"order_id": order_id, "today": today})
    if check.get("status") == "already_returned":
        return {"status": "return_started", "order_id": check["order_id"], "return_id": check["return_id"]}
    if not check.get("eligible"):
        return check
    return_id = f"RT-{len(RETURNS) + 1:04d}"
    RETURNS[check["order_id"]] = return_id
    return {"status": "return_started", "order_id": check["order_id"], "return_id": return_id, "reason": reason}
