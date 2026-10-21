"""Unit tests for the tools. No LLM or embedding API calls: RAG uses fake embeddings."""
from datetime import date

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.utils.function_calling import convert_to_openai_tool

from agent import rag, tools
from agent.session import Session

TODAY = date(2026, 10, 21)


class FrozenDate(date):
    @classmethod
    def today(cls):
        return TODAY


@pytest.fixture(autouse=True)
def clean_returns(monkeypatch):
    monkeypatch.setattr(tools, "date", FrozenDate)  # tests must not depend on the real date
    tools.RETURNS.clear()


# ---------- get_bike_spec ----------

def test_spec_differs_between_years():
    old = tools.get_bike_spec.invoke({"model": "Strada GX", "year": 2023})
    new = tools.get_bike_spec.invoke({"model": "strada gx", "year": 2025})
    assert old["max_rotor_front_mm"] == 160
    assert new["max_rotor_front_mm"] == 180


def test_unknown_model_lists_known_models():
    result = tools.get_bike_spec.invoke({"model": "Strada GZ", "year": 2025})
    assert result["error"] == "unknown_model"
    assert "Strada GX" in result["known_models"]


def test_unknown_year():
    result = tools.get_bike_spec.invoke({"model": "Strada GX", "year": 2024})
    assert result == {"error": "unknown_year", "year": 2024, "known_years": [2023, 2025]}


# ---------- check_compatibility ----------

def test_rotor_compatibility_depends_on_year():
    args = {"part_type": "front_rotor", "part_value": "180mm", "model": "Strada GX"}
    assert tools.check_compatibility.invoke({**args, "year": 2023})["compatible"] is False
    assert tools.check_compatibility.invoke({**args, "year": 2025})["compatible"] is True


def test_rim_brake_bike_has_no_rotor():
    result = tools.check_compatibility.invoke(
        {"part_type": "front_rotor", "part_value": "160", "model": "Metro C", "year": 2023})
    assert result["compatible"] is False
    assert "rim" in result["reason"]


def test_unknown_part_type():
    result = tools.check_compatibility.invoke(
        {"part_type": "saddle", "part_value": "160", "model": "Metro C", "year": 2025})
    assert result["error"] == "unknown_part_type"


# ---------- get_order_status ----------

def test_order_status_returns_only_status_and_dates():
    result = tools.get_order_status.invoke({"order_id": "cn-1044"})
    assert result == {"order_id": "CN-1044", "status": "shipped", "order_date": "2026-10-16", "delivery_date": None}


@pytest.mark.parametrize("order_id,error", [("1042", "invalid_order_id"), ("CN-9999", "order_not_found")])
def test_order_errors(order_id, error):
    assert tools.get_order_status.invoke({"order_id": order_id})["error"] == error


# ---------- search_support_docs / superseded filter ----------

@pytest.fixture
def fake_store(tmp_path):
    store = rag.build_index(embeddings=DeterministicFakeEmbedding(size=64), persist_dir=tmp_path)
    rag.set_store(store)
    yield store
    rag.set_store(None)


def test_superseded_returns_policy_is_filtered(fake_store):
    results = rag.search("returns policy", today=TODAY, k=10)
    sources = {r["source"] for r in results}
    assert "returns_policy_2026.md" in sources
    assert "returns_policy_2023.md" not in sources


def test_old_policy_was_valid_in_2024(fake_store):
    sources = {r["source"] for r in rag.search("returns policy", today=date(2024, 6, 1), k=10)}
    assert "returns_policy_2023.md" in sources
    assert "returns_policy_2026.md" not in sources


def test_model_filter_excludes_other_model_docs(fake_store):
    sources = {r["source"] for r in rag.search("care", model="Aero R1", today=TODAY, k=10)}
    assert "care_ridge_e.md" not in sources
    assert "care_general.md" in sources


def test_search_tool_rejects_unknown_model():
    assert tools.search_support_docs.invoke({"query": "x", "model": "Nope"})["error"] == "unknown_model"


# ---------- start_return: confirmation gate, window, idempotency ----------

def test_start_return_only_creates_pending_action():
    session = Session("s1")
    result = tools.start_return.invoke({"order_id": "CN-1042", "reason": "too small", "session": session})
    assert result["status"] == "confirmation_required"
    assert session.pending_action["order_id"] == "CN-1042"
    assert tools.RETURNS == {}  # nothing executed


def test_start_return_asks_reason_only_when_eligible():
    session = Session("s1b")
    result = tools.start_return.invoke({"order_id": "CN-1042", "session": session})
    assert result["status"] == "reason_required"
    assert session.pending_action is None


@pytest.mark.parametrize("order_id", ["CN-1043", "CN-1044"])  # outside window, not delivered
def test_start_return_rejects_ineligible_before_reason(order_id):
    session = Session("s1c")
    result = tools.start_return.invoke({"order_id": order_id, "session": session})
    assert result["eligible"] is False
    assert "reason_required" not in result.values()
    assert session.pending_action is None


def test_execute_without_pending_confirmation_fails():
    session = Session("s2")
    assert tools.execute_pending_return(session, today=TODAY) == {"error": "no_pending_confirmation"}
    assert tools.RETURNS == {}


def test_execute_after_confirmation():
    session = Session("s3")
    session.pending_action = {"type": "start_return", "order_id": "CN-1042", "reason": "too small"}
    result = tools.execute_pending_return(session, today=TODAY)
    assert result["status"] == "return_started"
    assert session.pending_action is None


def test_return_is_idempotent():
    ids = []
    for _ in range(2):
        session = Session("s4")
        session.pending_action = {"type": "start_return", "order_id": "CN-1042", "reason": "x"}
        ids.append(tools.execute_pending_return(session, today=TODAY)["return_id"])
    assert ids[0] == ids[1]
    assert len(tools.RETURNS) == 1


def test_return_window_inside():
    # CN-1042 delivered 2026-10-15
    assert tools.check_return_eligibility("CN-1042", today=date(2026, 11, 14))["eligible"] is True


def test_return_window_outside():
    result = tools.check_return_eligibility("CN-1042", today=date(2026, 11, 15))
    assert result["eligible"] is False
    assert "30-day" in result["reason"]


def test_return_not_delivered():
    assert tools.check_return_eligibility("CN-1044", today=TODAY)["eligible"] is False


# ---------- schemas ----------

def test_session_is_hidden_from_model_schema():
    schema = convert_to_openai_tool(tools.start_return)
    assert set(schema["function"]["parameters"]["properties"]) == {"order_id", "reason"}
