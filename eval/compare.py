"""Print a side-by-side table of eval summaries. Usage: python -m eval.compare v1 v2"""
import json
import sys
from pathlib import Path

versions = sys.argv[1:] or ["v1", "v2"]
runs = {v: json.loads((Path(__file__).parent / f"results_{v}.json").read_text()) for v in versions}

print(f"{'metric':<32}" + "".join(f"{v:>12}" for v in versions))
for metric in runs[versions[0]]["summary"]:
    print(f"{metric:<32}" + "".join(f"{str(runs[v]['summary'].get(metric, '')):>12}" for v in versions))

print("\nPer case")
for i, case in enumerate(runs[versions[0]]["cases"]):
    row = [("PASS" if runs[v]["cases"][i]["passed"] else "FAIL") for v in versions]
    failed = {v: [k for k, ok in runs[v]["cases"][i]["checks"].items() if not ok] for v in versions}
    detail = "  ".join(f"{v}:{failed[v]}" for v in versions if failed[v])
    print(f"{case['id']:<40}" + "".join(f"{x:>6}" for x in row) + f"  {detail}")
