"""Print a side-by-side table of eval summaries. Usage: python -m eval.compare v1 v2"""
import json
import sys
from pathlib import Path

versions = sys.argv[1:] or ["v1", "v2"]
runs = {v: json.loads((Path(__file__).parent / f"results_{v}.json").read_text()) for v in versions}

print(f"{'metric':<32}" + "".join(f"{v:>18}" for v in versions))
for metric in runs[versions[0]]["summary"]:
    print(f"{metric:<32}" + "".join(f"{str(runs[v]['summary'].get(metric, '')):>18}" for v in versions))

print("\nCases that failed at least once (passes / runs)")
for case_id in dict.fromkeys(c["id"] for c in runs[versions[0]]["cases"]):
    cells, failures = [], {}
    for v in versions:
        rs = [c for c in runs[v]["cases"] if c["id"] == case_id]
        cells.append(f"{sum(c['passed'] for c in rs)}/{len(rs)}")
        failures[v] = sorted({k for c in rs for k, ok in c["checks"].items() if not ok})
    if any(failures.values()):
        detail = "  ".join(f"{v}:{f}" for v, f in failures.items() if f)
        print(f"{case_id:<40}" + "".join(f"{x:>6}" for x in cells) + f"  {detail}")
