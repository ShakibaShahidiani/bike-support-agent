"""Aggregate the JSONL logs into monitoring metrics per prompt version.

Turns and feedback are joined on (session_id, turn_id). Thumbs-up rate is the CSAT proxy.
Eval runs log to the same file (session ids start with "eval-") and are excluded by default.
"""
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from agent import config

FALLBACK_PREFIXES = ("ungrounded:", "escalation_fallback", "model_error:", "max_iterations")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _pct(part: int, whole: int) -> float | None:
    return round(100 * part / whole, 1) if whole else None


def _p(values: list[int], q: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def compute_metrics(turns: list[dict], feedback: list[dict], include_eval: bool = False) -> dict:
    if not include_eval:
        turns = [t for t in turns if not t["session_id"].startswith("eval-")]
    # last rating wins if a user changes their mind
    ratings = {(f["session_id"], f["turn_id"]): f["rating"] for f in feedback}

    by_version = defaultdict(list)
    for t in turns:
        by_version[t["prompt_version"]].append(t)

    rows = []
    for version, ts in sorted(by_version.items()):
        rated = [ratings[(t["session_id"], t["turn_id"])] for t in ts if (t["session_id"], t["turn_id"]) in ratings]
        latencies = [t["latency_ms"]["total"] for t in ts]
        costs = [t["cost_usd"] for t in ts if t.get("cost_usd") is not None]
        flags = [f for t in ts for f in t["flags"]]
        rows.append({
            "prompt_version": version,
            "turns": len(ts),
            "sessions": len({t["session_id"] for t in ts}),
            "rated_turns": len(rated),
            "thumbs_up_rate": _pct(rated.count("up"), len(rated)),
            "latency_p50_ms": _p(latencies, 0.5),
            "latency_p95_ms": _p(latencies, 0.95),
            "avg_tokens": round(statistics.mean(t["input_tokens"] + t["output_tokens"] for t in ts)),
            "avg_cost_usd": round(statistics.mean(costs), 6) if costs else None,
            "total_cost_usd": round(sum(costs), 4) if costs else None,
            "escalation_rate": _pct(sum("safety_escalation" in t["flags"] for t in ts), len(ts)),
            "fallback_rate": _pct(sum(any(f.startswith(FALLBACK_PREFIXES) for f in t["flags"]) for t in ts), len(ts)),
            "returns_executed": flags.count("return_executed"),
        })

    tool_usage = Counter(c["name"] for t in turns for c in t["tools_called"])
    turn_index = {(t["session_id"], t["turn_id"]): t for t in turns}
    thumbs_down = [
        {"prompt_version": t["prompt_version"], "session_id": t["session_id"], "turn_id": t["turn_id"],
         "user_message": t["user_message"], "answer": t.get("answer", ""), "flags": t["flags"]}
        for key, rating in ratings.items() if rating == "down" and (t := turn_index.get(key))
    ]
    return {"by_version": rows, "tool_usage": dict(tool_usage.most_common()), "thumbs_down": thumbs_down}


def load_metrics(include_eval: bool = False) -> dict:
    return compute_metrics(read_jsonl(config.LOGS_DIR / "turns.jsonl"),
                           read_jsonl(config.LOGS_DIR / "feedback.jsonl"), include_eval)
