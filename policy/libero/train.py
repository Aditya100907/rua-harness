"""Train stock DEM (sentence format) on LIBERO demos, reusing `policy.train` unchanged.

`policy.train.main` is run with two substitutions: the dataset is `LiberoDataset` (agentview + wrist,
8-number state, 7-number action) and the policy is built with `DEMConfig(action_dim=7, proprio_dim=8)`.
Everything else (optimizer, schedule, bf16, checkpoints, --resume, --max-hours, norm_stats.json,
train_log.csv) is the RoboCasa code path. Runs in the `envs/libero` env (DEM is installed there) or
in `envs/dem` (needs only h5py and opencv; no simulator).

    python -m policy.libero.train --suites libero_spatial --steps 30000 --out RUN_DIR [policy.train args]
"""
from __future__ import annotations

import argparse
import sys

from policy.libero import sim


def _build_policy(fmt: str = "sentence", cams: int = 2, offline: bool = False, freeze_vision: bool = False):
    from dem import DEM, DEMConfig
    if fmt != "sentence":
        raise ValueError("LIBERO training supports only the sentence format so far")
    return DEM(DEMConfig(action_dim=sim.ACTION_DIM, proprio_dim=sim.STATE_DIM, chunk=16, num_cameras=cams,
                         freeze_vision=freeze_vision, vision_pretrained=not offline,
                         language_pretrained=not offline))


def _dataset(tasks, cams=sim.DEFAULT_CAMS, norm_stats=None, episodes=None, use_video=False):
    from policy.libero.data import LiberoDataset
    return LiberoDataset(tasks, cams=tuple(cams), norm_stats=norm_stats, episodes=episodes)


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--suites", nargs="+", default=["libero_spatial"])
    a, rest = ap.parse_known_args(argv)
    from policy.libero.data import suite_tasks
    tasks = [t for s in a.suites for t in suite_tasks(s)]
    if not tasks:
        raise FileNotFoundError(f"no LIBERO demos for {a.suites} under {sim.data_root()}")
    if "--cams" not in rest:
        rest += ["--cams", *sim.DEFAULT_CAMS]
    import policy.model
    import policy.train
    policy.model.build_policy = _build_policy
    policy.train.RoboCasaDataset = _dataset
    print(f"LIBERO training on {len(tasks)} tasks from {a.suites}", flush=True)
    policy.train.main(["--tasks", *tasks, *rest])


if __name__ == "__main__":
    main()
