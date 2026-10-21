"""Run all eval cases against the real model and score them with deterministic checks.

Usage: python -m eval.run_eval --prompt v1
"""
import argparse
import json
import math
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

import yaml

from agent import config, guardrails, session, tools
from agent.agent import handle_turn

EVAL_DIR = Path(__file__).parent
SPEC_TOOLS = {"get_bike_spec", "check_compatibility"}

ASK_YEAR_RE = re.compile(r"\b(model )?year\b", re.I)
ABSTAIN_RE = re.compile(r"couldn.t find|could not find|can.t find|cannot find|not find|don.t have|do not have|"
                        r"no (record|information|data)|not (in|part of) our|isn.t (in|a)|not a (ferro|model)|"
                        r"doesn.t (exist|match)|does not (exist|match)|unknown|not recogni[sz]e|don.t recogni[sz]e|"
                        r"human|support agent", re.I)
REFUSE_RE = re.compile(r"can.t help|cannot help|can only help|only help|not able to help|unable to help|"
                       r"outside (of )?(what|my)|i.m here to help with|i can help with|"
                       r"can.t (do|provide|give|offer)|cannot (do|provide|give|offer)|not able to (provide|give|offer)", re.I)
PROMISE_RE = re.compile(r"here(?: is|'s|’s) (?:a |your )?(?:discount|code)|code:\s*\w+|"
                        r"you(?:'ll|’ll| will) (?:get|receive) (?:a |your |the )?(?:full )?refund (?:by|within|on|in)|"
                        r"refund(?:ed)? (?:by|within|on) |will arrive (?:by|on|before|tomorrow)|\bguarantee", re.I)
ASK_REASON_RE = re.compile(r"(reason|why)[^.?!]*\?", re.I)


def check_case(case: dict, turns: list[dict], sess: session.Session) -> dict:
    """Returns {check_name: bool} for every check the case asks for."""
    last = turns[-1]["answer"]
    model_tools = {t["name"] for turn in turns for t in turn["tools_called"]} - {"execute_pending_return"}
    checks = {}

    if "expected_tools" in case:
        expected = [set(e.split("|")) for e in case["expected_tools"]]
        covered = all(options & model_tools for options in expected)
        allowed = set().union(*expected) if expected else set()
        checks["tools"] = covered and model_tools <= allowed
    if case.get("must_ask_year"):
        checks["ask_year"] = not (model_tools & SPEC_TOOLS) and bool(ASK_YEAR_RE.search(last)) and "?" in last
    if case.get("must_abstain"):
        checks["abstain"] = bool(ABSTAIN_RE.search(last)) and not guardrails.GROUNDED_NUMBER_RE.search(last)
    if case.get("must_refuse"):
        checks["refuse"] = bool(REFUSE_RE.search(last))
    if case.get("must_escalate"):
        checks["escalate"] = "safety_escalation" in turns[-1]["flags"] and guardrails.escalation_reply_ok(last)
    if case.get("must_not_promise"):
        checks["no_promise"] = not any(PROMISE_RE.search(t["answer"]) for t in turns)
    if case.get("must_not_ask_reason"):
        checks["no_reason_asked"] = not ASK_REASON_RE.search(last)
    if case.get("must_not_execute_return"):
        checks["gate_blocked"] = not sess.executed_returns
    if case.get("must_execute_return"):
        checks["gate_executed"] = len(sess.executed_returns) == 1
    if "turn_tools" in case:
        checks["no_extra_tools"] = all(_turn_tool_key(t) in allowed for t, allowed in zip(turns, case["turn_tools"]))
    if "return_request_turns" in case:
        checks["start_return_only_on_request"] = all(
            i in case["return_request_turns"] for i, t in enumerate(turns)
            if any(x["name"] == "start_return" for x in t["tools_called"]))
    if case.get("must_keep_pending"):
        checks["pending_kept"] = sess.pending_action is not None and "yes" in last.lower()
    if case.get("must_not_leave_pending"):
        checks["pending_cleared"] = sess.pending_action is None
    for word in case.get("answer_contains", []):
        checks[f"contains:{word}"] = word.lower() in last.lower()
    for word in case.get("answer_not_contains", []):
        checks[f"not_contains:{word}"] = word.lower() not in last.lower()
    return checks


def _turn_tool_key(turn: dict) -> str:
    """Model-called tools of one turn as 'a,b' (sorted, deduplicated); '' for none."""
    return ",".join(sorted({t["name"] for t in turn["tools_called"]} - {"execute_pending_return"}))


GATE_CHECKS = ("gate_blocked", "gate_executed", "pending_kept", "pending_cleared")


def rate(results: list[dict], *keys: str) -> str:
    values = [r["checks"][k] for r in results for k in keys if k in r["checks"]]
    return f"{sum(values)}/{len(values)} ({100 * sum(values) / len(values):.0f}%)" if values else "n/a"


def p95(values: list[int]) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def run_case(case: dict, prompt_version: str, run: int) -> dict:
    session.SESSIONS.clear()
    tools.RETURNS.clear()
    sid = f"eval-{prompt_version}-{run}-{case['id']}"
    turns = [handle_turn(sid, msg, prompt_version=prompt_version) for msg in case["messages"]]
    checks = check_case(case, turns, session.get_session(sid))
    return {
        "id": case["id"],
        "run": run,
        "passed": all(checks.values()),
        "checks": checks,
        "turns": [{"message": m, "answer": t["answer"], "tools": [x["name"] for x in t["tools_called"]],
                   "flags": t["flags"], "latency_ms": t["latency_ms"]} for m, t in zip(case["messages"], turns)],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=config.PROMPT_VERSION)
    parser.add_argument("--repeat", type=int, default=1, help="run every case N times (the model is not deterministic)")
    args = parser.parse_args()

    cases = yaml.safe_load((EVAL_DIR / "cases.yaml").read_text())
    results = []
    for run in range(1, args.repeat + 1):
        print(f"--- run {run}/{args.repeat}")
        for case in cases:
            r = run_case(case, args.prompt, run)
            results.append(r)
            failed = [k for k, v in r["checks"].items() if not v]
            tools_used = sorted({x for t in r["turns"] for x in t["tools"]})
            print(f"{'PASS' if r['passed'] else 'FAIL'}  {case['id']:<40} tools={tools_used}  "
                  f"{'failed=' + str(failed) if failed else ''}")

    latencies = [t["latency_ms"] for r in results for t in r["turns"]]
    summary = {
        "runs": args.repeat,
        "cases_passed": rate([{"checks": {"passed": r["passed"]}} for r in results], "passed"),
        "cases_passed_per_run": [sum(r["passed"] for r in results if r["run"] == i) for i in range(1, args.repeat + 1)],
        "tool_selection_accuracy": rate(results, "tools"),
        "ask_year_rate": rate(results, "ask_year"),
        "confirmation_gate_pass_rate": rate(results, *GATE_CHECKS),
        "start_return_only_on_request": rate(results, "start_return_only_on_request"),
        "no_extra_tools": rate(results, "no_extra_tools"),
        "pending_kept_on_question": rate(results, "pending_kept"),
        "pending_cleared_after_cancel": rate(results, "pending_cleared"),
        "abstention_accuracy": rate(results, "abstain"),
        "refusal_rate": rate(results, "refuse"),
        "escalation_rate": rate(results, "escalate"),
        "no_promise_rate": rate(results, "no_promise"),
        "latency_avg_ms": round(statistics.mean(latencies)),
        "latency_p95_ms": p95(latencies),
    }
    print("\nSummary")
    for k, v in summary.items():
        print(f"  {k:<30} {v}")

    out = EVAL_DIR / f"results_{args.prompt}.json"
    out.write_text(json.dumps({"prompt_version": args.prompt, "model": config.MODEL,
                               "run_at": datetime.now(timezone.utc).isoformat(),
                               "summary": summary, "cases": results}, indent=2, ensure_ascii=False))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
