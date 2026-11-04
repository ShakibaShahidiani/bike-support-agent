"""The graph end to end with a scripted fake model (no API calls). The model's labels are given; the routing, slots,
confirmation pause and templates are the real code.
"""
from datetime import date
from types import SimpleNamespace

import pytest

from agent import agent, config, graph, llm, session, tools


@pytest.fixture
def model(monkeypatch, tmp_path):
    """Set `model.intent`, `model.parts` etc. to choose what the fake model answers."""
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(tools, "date", SimpleNamespace(today=lambda: date(2026, 10, 21), fromisoformat=date.fromisoformat))
    tools.RETURNS.clear()
    session.SESSIONS.clear()
    agent.reset()
    fake = SimpleNamespace(intent="other", parts={}, ret={}, label="other", doc=("", []), fail=set())

    def structured(schema, system, user):
        name = schema.__name__
        if name in fake.fail:
            raise RuntimeError("model down")
        return {
            "Route": lambda: graph.Route(intent=fake.intent),
            "PartsSlots": lambda: graph.PartsSlots(**{**dict.fromkeys(graph.PartsSlots.model_fields), **fake.parts}),
            "ReturnSlots": lambda: graph.ReturnSlots(**{**dict.fromkeys(graph.ReturnSlots.model_fields), **fake.ret}),
            "ConfirmLabel": lambda: graph.ConfirmLabel(label=fake.label),
            "Query": lambda: graph.Query(query="rewritten"),
            "DocAnswer": lambda: graph.DocAnswer(answer=fake.doc[0], used_chunks=fake.doc[1]),
        }[name]()

    monkeypatch.setattr(llm, "structured", structured)
    return fake


def _start_return(model, sid="s"):
    model.intent, model.ret = "return", {"order_id": "CN-1042", "reason": "too small"}
    return agent.handle_turn(sid, "I want to return CN-1042, it's too small")


# ---------- confirmation pause ----------

def test_return_pauses_and_executes_only_after_confirm(model):
    first = _start_return(model)
    assert first["pending_confirmation"] is True
    assert "CN-1042" in first["answer"]
    assert tools.RETURNS == {}

    model.label = "confirm"
    second = agent.handle_turn("s", "yes")
    assert "return_executed" in second["flags"]
    assert list(tools.RETURNS) == ["CN-1042"]
    assert second["pending_confirmation"] is False
    assert tools.RETURNS["CN-1042"] in second["answer"]


def test_cancel_changes_nothing(model):
    _start_return(model)
    model.label = "cancel"
    out = agent.handle_turn("s", "no")
    assert "cancelled" in out["answer"]
    assert tools.RETURNS == {} and out["pending_confirmation"] is False


def test_other_reply_cancels_then_goes_through_intake(model):
    _start_return(model)
    model.label, model.intent = "other", "human"
    out = agent.handle_turn("s", "my frame just cracked")
    assert "did not start the return" in out["answer"]
    assert "human support" in out["answer"]
    assert tools.RETURNS == {} and out["pending_confirmation"] is False


def test_too_long_reply_during_pause_skips_the_model(model):
    _start_return(model)
    model.fail = {"ConfirmLabel", "Route"}
    out = agent.handle_turn("s", "x" * 2001)
    assert "too long" in out["answer"]
    assert tools.RETURNS == {} and out["pending_confirmation"] is False


def test_model_failure_during_pause_never_executes(model):
    _start_return(model)
    model.fail = {"ConfirmLabel"}
    out = agent.handle_turn("s", "yes")
    assert "did not start" in out["answer"]
    assert tools.RETURNS == {} and out["pending_confirmation"] is False


def test_window_closing_during_pause_blocks_execution(model, monkeypatch):
    _start_return(model)
    monkeypatch.setattr(tools, "date", SimpleNamespace(today=lambda: date(2026, 12, 1), fromisoformat=date.fromisoformat))
    model.label = "confirm"
    out = agent.handle_turn("s", "yes")
    assert "return_failed" in out["flags"]
    assert tools.RETURNS == {}


def test_second_return_reports_the_same_id(model):
    _start_return(model)
    model.label = "confirm"
    agent.handle_turn("s", "yes")
    out = _start_return(model)
    assert tools.RETURNS["CN-1042"] in out["answer"]
    assert out["pending_confirmation"] is False
    assert len(tools.RETURNS) == 1


# ---------- return_check ----------

def test_missing_reason_is_asked_after_eligibility_then_continues(model):
    model.intent, model.ret = "return", {"order_id": "CN-1042"}
    first = agent.handle_turn("s", "return CN-1042")
    assert "reason" in first["answer"] and first["pending_confirmation"] is False
    model.intent, model.ret = "continue", {"reason": "too big"}
    assert agent.handle_turn("s", "it's too big")["pending_confirmation"] is True


@pytest.mark.parametrize("order_id", ["CN-1043", "CN-1044"])  # outside the window, not delivered
def test_ineligible_order_never_pauses(model, order_id):
    model.intent, model.ret = "return", {"order_id": order_id, "reason": "x"}
    out = agent.handle_turn("s", f"return {order_id}")
    assert "can't start a return" in out["answer"]
    assert out["pending_confirmation"] is False


# ---------- parts ----------

def test_year_is_asked_then_filled(model):
    model.intent, model.parts = "parts", {"model": "Strada GX", "attribute": "tyre_clearance_mm", "part_value": "45"}
    first = agent.handle_turn("s", "Will 45mm tyres fit my Strada GX?")
    assert "model year" in first["answer"] and first["tools_called"] == []

    model.intent, model.parts = "continue", {"year": 2023}
    second = agent.handle_turn("s", "2023")
    assert "No, that will not fit" in second["answer"] and "42mm" in second["answer"]


def test_bike_carries_over_to_the_next_question(model):
    model.intent, model.parts = "parts", {"model": "Aero R1", "year": 2025, "attribute": "tyre_clearance_mm"}
    agent.handle_turn("s", "tyre clearance on my 2025 Aero R1?")
    model.parts = {"attribute": "max_rotor_rear_mm"}
    out = agent.handle_turn("s", "and the rear rotor?")
    assert "2025 Aero R1" in out["answer"] and "160mm" in out["answer"]


def test_new_task_clears_the_bike(model):
    model.intent, model.parts = "parts", {"model": "Aero R1", "year": 2025, "attribute": "tyre_clearance_mm"}
    agent.handle_turn("s", "tyre clearance on my 2025 Aero R1?")
    model.intent = "order"
    agent.handle_turn("s", "where's CN-1044?")
    model.intent, model.parts = "parts", {"attribute": "battery_wh"}
    assert "Which Ferro Cycles model" in agent.handle_turn("s", "what's the battery?")["answer"]


def test_unknown_model_lists_catalogue(model):
    model.intent, model.parts = "parts", {"model": "Terra X", "year": 2025, "attribute": "tyre_clearance_mm"}
    out = agent.handle_turn("s", "tyre clearance on a 2025 Terra X?")
    assert "couldn't find" in out["answer"] and "Strada GX" in out["answer"] and "human" in out["answer"]
    assert out["tools_called"] == []


def test_unknown_year_lists_years(model):
    model.intent, model.parts = "parts", {"model": "Ridge E", "year": 2024, "attribute": "battery_wh"}
    out = agent.handle_turn("s", "battery on a 2024 Ridge E?")
    assert "2023 or 2025" in out["answer"] and out["tools_called"] == []


def test_comparison_answers_one_year_and_offers_the_other(model):
    model.intent, model.parts = "parts", {"model": "Ridge E", "year": 2023, "other_year": 2025, "attribute": "battery_wh"}
    out = agent.handle_turn("s", "battery 2023 vs 2025?")
    assert "625Wh" in out["answer"] and "750" not in out["answer"] and "2025" in out["answer"]


def test_missing_attribute_value_says_so(model):
    model.intent, model.parts = "parts", {"model": "Aero R1", "year": 2025, "attribute": "battery_wh"}
    assert "has no battery" in agent.handle_turn("s", "battery on my 2025 Aero R1?")["answer"]


# ---------- order status ----------

def test_order_status_from_data(model):
    model.intent = "order"
    out = agent.handle_turn("s", "where's cn-1044?")
    assert "shipped" in out["answer"] and out["tools_called"][0]["name"] == "get_order_status"


def test_order_id_is_asked_then_used(model):
    model.intent = "order"
    assert "order ID" in agent.handle_turn("s", "where's my bike?")["answer"]
    model.intent = "continue"
    assert "cancelled" in agent.handle_turn("s", "CN-1046")["answer"]


def test_unknown_order(model):
    model.intent = "order"
    assert "couldn't find order CN-9999" in agent.handle_turn("s", "CN-9999?")["answer"]


# ---------- docs ----------

@pytest.fixture
def chunks(monkeypatch):
    results = [{"source": "warranty.md", "content": "Frames: 5 years."}, {"source": "care_general.md", "content": "Rinse it."}]
    monkeypatch.setattr(tools.rag, "search", lambda *a, **k: results)


def test_docs_cites_only_chunks_it_used(model, chunks):
    model.intent, model.doc = "docs", ("Frames have a 5 years warranty.", [1, 1, 99])
    out = agent.handle_turn("s", "frame warranty?")
    assert out["answer"].endswith("Sources: warranty.md")


def test_docs_without_valid_citation_says_it_does_not_know(model, chunks):
    model.intent, model.doc = "docs", ("Frames last forever.", [99])
    assert agent.handle_turn("s", "frame warranty?")["answer"] == graph.DONT_KNOW


def test_docs_invented_number_is_blocked(model, chunks):
    model.intent, model.doc = "docs", ("You get 90 days.", [1])
    out = agent.handle_turn("s", "frame warranty?")
    assert "ungrounded" in out["flags"][-1] and "90" not in out["answer"]


# ---------- intake and fixed replies ----------

def test_too_long_makes_no_model_call(model):
    model.fail = {"Route"}
    out = agent.handle_turn("s", "x" * 2001)
    assert "too long" in out["answer"]
    assert "input_too_long" in out["flags"]


@pytest.mark.parametrize("intent,expected", [("human", "human support team"), ("other", "can't help")])
def test_fixed_replies(model, intent, expected):
    model.intent = intent
    assert expected in agent.handle_turn("s", "hello")["answer"]


def test_continue_with_no_task_is_other(model):
    model.intent = "continue"
    assert "can't help" in agent.handle_turn("s", "2023")["answer"]


def test_router_failure_gets_friendly_reply_and_the_next_turn_works(model):
    model.fail = {"Route"}
    assert "technical trouble" in agent.handle_turn("s", "hello")["answer"]
    model.fail, model.intent = set(), "other"
    assert "can't help" in agent.handle_turn("s", "hello")["answer"]


def test_failed_node_does_not_leave_the_graph_stuck(model):
    model.intent, model.fail = "parts", {"PartsSlots"}
    assert "technical trouble" in agent.handle_turn("s", "rotor?")["answer"]
    model.fail, model.intent = set(), "order"
    assert "shipped" in agent.handle_turn("s", "CN-1044?")["answer"]
