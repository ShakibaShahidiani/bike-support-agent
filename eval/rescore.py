"""Score the saved answers again with the current checks, without calling the model.

Shows how many old passes only passed because the scorer was too loose. Session-state checks can't be
recomputed from a saved answer, so those keep their saved value. The results files are not changed.

Usage: python -m eval.rescore v1 v2 v3
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

from eval.run_eval import GATE_CHECKS, check_case

EVAL_DIR = Path(__file__).parent


def rescore(saved_case: dict, case: dict) -> dict:
    turns = [{"answer": t["answer"], "flags": t["flags"], "tools_called": [{"name": n} for n in t["tools"]]}
             for t in saved_case["turns"]]
    checks = check_case(case, turns, SimpleNamespace(executed_returns=[], pending_action=None))
    return checks | {k: saved_case["checks"][k] for k in GATE_CHECKS if k in checks}


def main():
    cases = {c["id"]: c for c in yaml.safe_load((EVAL_DIR / "cases.yaml").read_text())}
    for version in sys.argv[1:] or ["v1", "v2", "v3"]:
        saved = json.loads((EVAL_DIR / f"results_{version}.json").read_text())["cases"]
        total = len(saved)
        old = new = 0
        changes = []
        for c in saved:
            checks = rescore(c, cases[c["id"]])
            passed = all(checks.values())
            old += c["passed"]
            new += passed
            if passed != c["passed"]:
                bad = [k for k, ok in checks.items() if not ok]
                changes.append(f"  run {c['run']} {c['id']}: {'FAIL' if c['passed'] else 'PASS'} now {bad or ''}\n"
                               f"    answer: {c['turns'][-1]['answer'][:200]}")
        print(f"{version}: saved {old}/{total} -> rescored {new}/{total}")
        print("\n".join(changes))


if __name__ == "__main__":
    main()
