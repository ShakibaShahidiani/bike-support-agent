"""Unit tests for the tools. No LLM or embedding API calls: RAG uses fake embeddings."""
from datetime import date

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from agent import rag, tools

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


# ---------- returns: window, idempotency ----------

def _eligibility(order_id, today=TODAY):
    return tools.check_return_eligibility.invoke({"order_id": order_id, "today": today})


def _create(order_id="CN-1042", today=TODAY):
    return tools.create_return.invoke({"order_id": order_id, "reason": "too small", "today": today})


def test_eligibility_check_changes_nothing():
    assert _eligibility("CN-1042")["eligible"] is True
    assert tools.RETURNS == {}


def test_return_window_inside():
    # CN-1042 delivered 2026-10-15
    assert _eligibility("CN-1042", date(2026, 11, 14))["eligible"] is True


def test_return_window_outside():
    result = _eligibility("CN-1042", date(2026, 11, 15))
    assert result["eligible"] is False
    assert "30-day" in result["reason"]


def test_return_not_delivered():
    assert _eligibility("CN-1044")["eligible"] is False


def test_eligibility_unknown_order():
    assert _eligibility("CN-9999")["error"] == "order_not_found"


def test_create_return():
    result = _create()
    assert result["status"] == "return_started"
    assert list(tools.RETURNS) == ["CN-1042"]


def test_create_return_rechecks_eligibility():
    result = _create(today=date(2026, 11, 15))  # the window closed while the customer was deciding
    assert result["eligible"] is False
    assert tools.RETURNS == {}


def test_return_is_idempotent():
    assert _create()["return_id"] == _create()["return_id"]
    assert len(tools.RETURNS) == 1


def test_already_returned_order_is_reported_with_its_id():
    return_id = _create()["return_id"]
    result = _eligibility("CN-1042")
    assert result["status"] == "already_returned"
    assert result["return_id"] == return_id
