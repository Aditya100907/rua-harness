"""Training labels that match the test-time "perfect" command exactly (fixes the label/test mismatch).

The original hindsight labels (policy/label_commands.py) put the grasp point at the gripper-close position
and the place point at the gripper-open position. At test time, policy/commands_gt.perfect_command uses
the object's body position and the destination's centre (or the task part nearest the gripper). The two
differ by 2 to 6 cm median, up to about 10 cm. Here every demo is reset to its frame 0 and labelled with
perfect_command itself, so training and test use one definition.

Output: <labels_base>_center/<task>.jsonl, same schema as the original labels; the original points are
kept as grasp_world_hindsight / place_world_hindsight for comparison.
    python -m policy.relabel_center --task PickPlaceCounterToCabinet [--chunk 0 --chunk-size 100]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from policy import paths, sim
from policy.commands_gt import perfect_command
from policy.label_commands import load_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=100000)
    ap.add_argument("--labels", default=None)
    a = ap.parse_args()
    src = Path(a.labels) if a.labels else paths.frames_base().parent / "labels"
    dst = Path(str(src) + "_center")
    dst.mkdir(parents=True, exist_ok=True)
    labels = load_labels(a.task, src)
    sel = labels[a.chunk * a.chunk_size:(a.chunk + 1) * a.chunk_size]
    ds = paths.task_dir(a.task)
    env = sim.make_env(ds, cams=None)
    out = dst / (f"{a.task}.part{a.chunk:03d}.jsonl" if a.chunk_size < 100000 else f"{a.task}.jsonl")
    shifts = []
    with open(out, "w") as f:
        for lb in sel:
            sim.reset_to_episode(env, ds, lb["episode"])
            cmd = perfect_command(env, a.task)
            new = dict(lb)
            new["grasp_world_hindsight"], new["place_world_hindsight"] = lb.get("grasp_world"), lb.get("place_world")
            g = cmd["grasp_world"] if cmd["has"][0] else None
            p = cmd["place_world"] if cmd["has"][1] else None
            new["grasp_world"] = None if g is None else [round(float(v), 5) for v in g]
            new["place_world"] = None if p is None else [round(float(v), 5) for v in p]
            new["kv_text_check"] = cmd["kv_text"]
            new["method"] = "perfect_command_frame0"
            if g is not None and lb.get("grasp_world") is not None:
                shifts.append(float(np.linalg.norm(np.asarray(g) - np.asarray(lb["grasp_world"]))))
            f.write(json.dumps(new) + "\n")
    env.close()
    if shifts:
        s = np.asarray(shifts) * 100
        print(f"RELABEL {a.task} chunk {a.chunk}: {len(sel)} demos; grasp shift vs hindsight cm: "
              f"median {np.median(s):.1f}, p95 {np.percentile(s, 95):.1f}, max {s.max():.1f}", flush=True)
    else:
        print(f"RELABEL {a.task} chunk {a.chunk}: {len(sel)} demos", flush=True)


if __name__ == "__main__":
    main()
