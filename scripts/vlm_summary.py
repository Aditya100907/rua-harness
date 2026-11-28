"""Open-model commander results next to perfect commands for the same checkpoints and seeds.

    python scripts/vlm_summary.py --formats sentence points_2d --episodes 10

Reads results/*_agent_plus_policy_vlm_qwen3vl8b_fs_mse_<format>_20000_<task>.csv and the perfect-command rows
(results/*_policy_only_fs_mse_<format>_20000_<task>.csv, seeds seed0 .. seed0+episodes-1). Prints a Markdown table.
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import statistics as st
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"
TASKS = ["PickPlaceCounterToCabinet", "OpenDrawer", "TurnOnMicrowave"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


def rows(pattern):
    out = []
    for f in sorted(glob.glob(str(R / pattern))):
        out += list(csv.DictReader(open(f)))
    return out


def fmt_rate(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({100 * k / n:.0f}%, {100 * lo:.0f} to {100 * hi:.0f})" if n else "none"


def med(xs):
    xs = [float(x) for x in xs if x not in ("", None)]
    return (f"{st.median(xs):.1f}", len(xs)) if xs else ("", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formats", nargs="+", default=["sentence", "points_2d"])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--prefix", default="qwen3vl8b")
    a = ap.parse_args()
    seeds = set(range(a.seed0, a.seed0 + a.episodes))
    print("| format | task | perfect commands | open-model commander | grasp err cm (median, n) | place err cm (median, n) "
          "| wall s / ep | model calls / ep | prompt tok / ep | completion tok / ep | skills / ep |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for fm in a.formats:
        for task in TASKS + ["all"]:
            ts = TASKS if task == "all" else [task]
            v, p = [], []
            for t in ts:
                v += [r for r in rows(f"*_agent_plus_policy_vlm_{a.prefix}_fs_mse_{fm}_20000_{t}.csv")
                      if int(r["seed"]) in seeds]
                pr = {int(r["seed"]): r for r in rows(f"*_policy_only_fs_mse_{fm}_20000_{t}.csv")
                      if int(r["seed"]) in seeds}
                p += list(pr.values())
            kv, nv = sum(int(r["success"]) for r in v), len(v)
            kp, np_ = sum(int(r["success"]) for r in p), len(p)
            g, ng = med([r["grasp_err_cm"] for r in v])
            pl, npl = med([r["place_err_cm"] for r in v])
            mean = lambda k: f"{st.mean(float(r[k]) for r in v):.0f}" if v else ""
            print(f"| {fm} | {task} | {fmt_rate(kp, np_)} | {fmt_rate(kv, nv)} | {g} ({ng}) | {pl} ({npl}) | "
                  f"{mean('seconds')} | {mean('llm_calls')} | {mean('input_tokens')} | {mean('output_tokens')} | "
                  f"{mean('n_skill')} |")


if __name__ == "__main__":
    main()
