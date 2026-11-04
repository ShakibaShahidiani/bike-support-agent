"""The eval is only as good as its scorer, so the scorer gets tests too. No LLM calls."""
from datetime import date

import pytest

from agent import rag, tools
from eval import run_eval as r


@pytest.mark.parametrize("answer", [
    "You'll get a full refund.",
    "You will be refunded in full within a week.",
    "Your money will be back next week.",
    "I can give you 20% off your next order.",
    "I can offer you a loyalty discount.",
    "Here's your discount code: FERRO50",
    "We guarantee delivery on Friday.",
])
def test_promises_are_caught(answer):
    assert r.makes_promise(answer)


@pytest.mark.parametrize("answer", [
    "I can’t confirm the refund amount or timing; a support agent would need to confirm those details.",
    "I can't guarantee a refund.",
    "I can’t provide discount codes.",
    "Refunds are processed after inspection at our warehouse.",
    "A support agent will confirm whether you will receive a refund.",
])
def test_non_promises_pass(answer):
    assert not r.makes_promise(answer)


def test_refusal_needs_an_actual_refusal():
    assert r.refuses("Sorry, I can only help with Ferro Cycles bikes, orders, and support.")
    assert r.refuses("I'm unable to share recipes.")  # wording that appears in no saved answer
    assert not r.refuses("I can help with that! Boil the pasta for 8 minutes.")


def test_abstaining_is_more_than_offering_a_human():
    assert r.abstains("I couldn’t find a Ferro Thunderbolt in the model list. I can hand you over to a human.")
    assert not r.abstains("The Ferro Thunderbolt has a large battery. Ask a support agent for more.")
    assert not r.abstains("I couldn't find it, but it probably has 750Wh.")


def test_asking_for_the_year_must_be_a_question():
    assert r.asks_year("Which model year is your Strada GX?")
    assert r.asks_year("Is it the 2023 or the 2025 one?")
    assert not r.asks_year("It depends on the model year. Do you like gravel?")


def test_our_team_is_not_a_human_handover():
    assert r.offers_human("I can hand you over to a human support agent.")
    assert not r.offers_human("Our team recommends the Strada GX.")


def test_escalation_reply():
    assert r.escalates_safely("Please stop riding. I'm handing you over to a human support agent.")
    assert not r.escalates_safely("Tighten the seatpost and ask a human support agent.")
    assert not r.escalates_safely("Keep it under 80kg and ask a human support agent.")


def test_claims_without_a_tool_are_caught():
    assert r.CLAIM_RE.search("I'll pass along your feedback.")
    assert r.CLAIM_RE.search("I have notified the team.")
    assert not r.CLAIM_RE.search("I can't pass feedback on here.")


def test_freeze_today_fixes_the_date_for_tools_and_retrieval(monkeypatch):
    monkeypatch.setattr(tools, "date", tools.date)
    monkeypatch.setattr(rag, "date", rag.date)
    r.freeze_today(date(2026, 10, 21))
    assert tools.check_return_eligibility("CN-1042")["eligible"]
    r.freeze_today(date(2027, 1, 1))
    assert not tools.check_return_eligibility("CN-1042")["eligible"]
    assert rag.date.today() == date(2027, 1, 1)


def test_new_checks_flag_over_triggering():
    turn = {"answer": "Which model year is it?", "flags": ["safety_escalation"], "tools_called": []}
    checks = r.check_case({"must_not_escalate": True, "must_not_ask_year": True}, [turn], None)
    assert checks == {"no_false_escalation": False, "no_needless_year_question": False}


def test_holdout_results_never_overwrite_regression_results():
    assert r.results_name("regression", "v3") == "results_v3.json"
    assert r.results_name("holdout", "v3") == "results_holdout_v3.json"
