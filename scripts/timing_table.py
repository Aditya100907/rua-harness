"""H2: wall-clock seconds per episode for the policy alone, split into simulator and model time.

Reads results/*policy_only*.csv rows that have `model_seconds` (written by policy/eval.py) and prints a
markdown table per run and task: episodes, mean steps, mean seconds, model seconds, simulator seconds
(the rest: physics, rendering, env bookkeeping), and seconds per robot second (20 steps = 1 s).
    python scripts/timing_table.py [results_dir]
"""
import csv
import glob
import sys
from collections import defaultdict

d = sys.argv[1] if len(sys.argv) > 1 else "results"
agg = defaultdict(list)
for f in glob.glob(f"{d}/*policy_only*.csv"):
    for r in csv.DictReader(open(f)):
        if r.get("model_seconds") in (None, "") or r.get("run_id") in ("wiring", ""):
            continue
        agg[(r["run_id"], r.get("checkpoint_step", ""), r["task"])].append(
            (float(r["seconds"]), float(r["model_seconds"]), int(r["steps"])))
print("| run | step | task | episodes | mean steps | s/episode | model s | sim s | wall s per robot s |")
print("|---|---|---|---|---|---|---|---|---|")
for (run, step, task), v in sorted(agg.items()):
    s = sum(x[0] for x in v) / len(v); m = sum(x[1] for x in v) / len(v); n = sum(x[2] for x in v) / len(v)
    print(f"| {run} | {step} | {task} | {len(v)} | {n:.0f} | {s:.1f} | {m:.1f} | {s - m:.1f} | {s / (n / 20):.2f} |")
