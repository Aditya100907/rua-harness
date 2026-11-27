"""One random-action LIBERO episode, headless. Checks env build, rendering and stepping.

    MUJOCO_GL=egl python -m policy.libero.random_episode --suite libero_spatial --task 0 --steps 50
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from policy.libero import sim


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--save", default="random_episode_frames.png")
    args = ap.parse_args()
    try:
        import OpenGL.GL as gl  # noqa: F401
    except Exception:
        pass
    t0 = time.time()
    info = sim.task_info(args.suite, args.task)
    print("task:", info["name"], "|", info["language"], "| init states:", tuple(info["init_states"].shape))
    env = sim.make_env(info["bddl"])
    obs = sim.reset_to_state(env, info["init_states"][0])
    print(f"built and reset in {time.time() - t0:.1f}s; obs keys:",
          {k: tuple(np.shape(v)) for k, v in obs.items() if k.startswith("robot0_") or k.endswith("image")})
    rng = np.random.default_rng(0)
    t1 = time.time()
    first = sim.images(obs)
    for _ in range(args.steps):
        a = rng.uniform(-1, 1, 7)
        a[6] = 1.0 if a[6] > 0 else -1.0
        obs, r, done, _ = env.step(a)
    print(f"{args.steps} random steps in {time.time() - t1:.1f}s, success={env.check_success()}, "
          f"state8={np.round(sim.state8(obs), 3).tolist()}")
    last = sim.images(obs)
    print("image mean (agentview, wrist):", [round(float(x.mean()), 1) for x in last])
    import imageio
    imageio.imwrite(args.save, np.concatenate([np.concatenate(list(first), 1), np.concatenate(list(last), 1)], 0))
    print("saved", Path(args.save).resolve())
    env.close()
    print("RANDOM EPISODE OK")


if __name__ == "__main__":
    main()
