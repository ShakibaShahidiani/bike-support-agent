"""Metrics aggregation over hand-made log lines (no API calls)."""
from agent.metrics import compute_metrics


def _turn(sid, tid, version="v3", latency=1000, flags=(), tools=(), cost=0.0001):
    return {"session_id": sid, "turn_id": tid, "prompt_version": version, "user_message": f"msg {tid}",
            "answer": f"answer {tid}", "latency_ms": {"total": latency}, "input_tokens": 100,
            "output_tokens": 20, "cost_usd": cost, "flags": list(flags),
            "tools_called": [{"name": n, "args": {}} for n in tools]}


TURNS = [
    _turn("a", 1, latency=1000, tools=["get_bike_spec"]),
    _turn("a", 2, latency=3000, flags=["safety_escalation"]),
    _turn("b", 1, latency=2000, flags=["ungrounded:203 mm"], tools=["get_bike_spec", "search_support_docs"]),
    _turn("b", 2, version="v2", latency=500),
    _turn("eval-v3-1-x", 1, latency=9000),
]
FEEDBACK = [
    {"session_id": "a", "turn_id": 1, "rating": "up"},
    {"session_id": "a", "turn_id": 2, "rating": "down"},
    {"session_id": "b", "turn_id": 1, "rating": "down"},
    {"session_id": "b", "turn_id": 1, "rating": "up"},  # changed mind: last rating wins
]


def test_groups_by_version_and_excludes_eval():
    m = compute_metrics(TURNS, FEEDBACK)
    v3 = next(r for r in m["by_version"] if r["prompt_version"] == "v3")
    assert [r["prompt_version"] for r in m["by_version"]] == ["v2", "v3"]
    assert v3["turns"] == 3 and v3["sessions"] == 2
    assert v3["latency_p50_ms"] == 2000 and v3["latency_p95_ms"] == 3000


def test_thumbs_up_rate_and_rates():
    v3 = next(r for r in compute_metrics(TURNS, FEEDBACK)["by_version"] if r["prompt_version"] == "v3")
    assert v3["rated_turns"] == 3
    assert v3["thumbs_up_rate"] == 66.7
    assert v3["escalation_rate"] == 33.3
    assert v3["fallback_rate"] == 33.3


def test_unrated_version_has_no_thumbs_rate():
    v2 = next(r for r in compute_metrics(TURNS, FEEDBACK)["by_version"] if r["prompt_version"] == "v2")
    assert v2["thumbs_up_rate"] is None


def test_include_eval():
    m = compute_metrics(TURNS, FEEDBACK, include_eval=True)
    assert sum(r["turns"] for r in m["by_version"]) == 5


def test_tool_usage_and_thumbs_down():
    m = compute_metrics(TURNS, FEEDBACK)
    assert m["tool_usage"] == {"get_bike_spec": 2, "search_support_docs": 1}
    assert [(t["session_id"], t["turn_id"]) for t in m["thumbs_down"]] == [("a", 2)]


def test_empty_logs():
    assert compute_metrics([], []) == {"by_version": [], "tool_usage": {}, "thumbs_down": []}
