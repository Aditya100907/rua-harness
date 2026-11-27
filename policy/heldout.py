"""Held-out object split for the pick-and-place tasks (F5).

Held-out categories are drawn with a fixed seed from the task's target-object
categories (`ep_meta` object `obj`, via the hindsight labels). A demo goes to
training only if no object in its scene (target or distractor) is held out, so
no held-out object appears in any training episode. Demos whose target object is
held out form the held-out evaluation pool, split in half: `val` (for choosing and
tuning) and `test` (locked: score only for results going into the paper, and log
each use in docs/heldout_split.md).

Evaluation on held-out objects replays those demos' initial scenes
(`policy/eval.py --demo-scenes <json> --demo-split val|test`).

    python -m policy.heldout --tasks PickPlaceCounterToCabinet --frac 0.15
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from policy import paths
from policy.label_commands import load_labels


def make_split(task: str, labels_dir: Path, frac: float = 0.15, seed: int = 0) -> dict:
    labels = load_labels(task, labels_dir)
    cats = sorted({lb["object_cat"] for lb in labels if lb["object_cat"]})
    rng = np.random.default_rng(seed)
    held = sorted(rng.choice(cats, size=max(1, int(round(frac * len(cats)))), replace=False).tolist())
    held_raw = {c.replace(" ", "_") for c in held}
    train, pool, dropped = [], [], []
    for lb in labels:
        scene = set(lb["objects"].values())
        if lb["object_cat"] in held:
            pool.append(lb["episode"])
        elif scene & held_raw:
            dropped.append(lb["episode"])        # a held-out object appears as a distractor
        else:
            train.append(lb["episode"])
    pool = sorted(pool)
    perm = np.random.default_rng(seed + 1).permutation(len(pool))
    val = sorted(pool[i] for i in perm[: len(pool) // 2])
    test = sorted(pool[i] for i in perm[len(pool) // 2:])
    # check: no held-out category anywhere in a training scene
    by_ep = {lb["episode"]: lb for lb in labels}
    leaks = [e for e in train if set(by_ep[e]["objects"].values()) & held_raw]
    assert not leaks, leaks
    return {"task": task, "seed": seed, "frac": frac, "n_categories": len(cats), "heldout_categories": held,
            "train": train, "val": val, "test": test, "dropped_distractor": dropped,
            "counts": {"train": len(train), "val": len(val), "test": len(test), "dropped": len(dropped),
                       "total": len(labels)}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["PickPlaceCounterToCabinet"])
    ap.add_argument("--frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--all-tasks-train", nargs="*", default=None,
                    help="also list these non-pick tasks with all their episodes in the training file")
    args = ap.parse_args()
    ld = Path(args.labels) if args.labels else paths.frames_base().parent / "labels"
    out = Path(args.out) if args.out else paths.frames_base().parent / "splits"
    out.mkdir(parents=True, exist_ok=True)
    train_eps, val, test = {}, {}, {}
    for t in args.tasks:
        s = make_split(t, ld, args.frac, args.seed)
        (out / f"heldout_{t}.json").write_text(json.dumps(s, indent=1))
        train_eps[t], val[t], test[t] = s["train"], s["val"], s["test"]
        print(t, s["counts"], "held out:", ", ".join(s["heldout_categories"]))
    for t in args.all_tasks_train or []:
        from policy.extract_frames import episode_lengths
        train_eps[t] = list(range(len(episode_lengths(paths.task_dir(t)))))
    (out / "objsplit_train.json").write_text(json.dumps(train_eps))
    (out / "objsplit_val.json").write_text(json.dumps(val))
    (out / "objsplit_test.json").write_text(json.dumps(test))
    print("wrote", out)


if __name__ == "__main__":
    main()
