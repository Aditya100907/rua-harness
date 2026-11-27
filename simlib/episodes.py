"""Example episode functions for simlib.runner. Copy one as the template for a real controller."""
from __future__ import annotations

import time

import numpy as np


def random_actions(task: str, seed: int, episode: int, max_steps: int | None = None,
                   scale: float = 1.0, size: int = 128, depth: bool = False) -> dict:
    """Uniform random actions until success or the task horizon. Expect about 0% success.

    Useful as a floor and to check the runner. No LLM, so token columns stay 0.
    """
    from simlib.tasks import check_success, horizon, make_env
    env = make_env(task, seed=seed, size=size, depth=depth)
    try:
        env.reset()
        t0 = time.time()   # time the rollout only, not env construction
        rng = np.random.default_rng(seed)
        lo, hi = env.action_spec
        steps = max_steps or horizon(task)
        success = False
        for t in range(steps):
            a = rng.uniform(lo, hi) * scale
            env.step(a)
            if check_success(env):
                success = True
                break
        print(f"{task} seed {seed}: success={success} after {t + 1} steps")
        return {"success": success, "steps": t + 1, "seconds": time.time() - t0}
    finally:
        env.close()


def noop(task: str, seed: int, episode: int, max_steps: int | None = None,
         size: int = 128) -> dict:
    """Zero actions until the horizon. Checks that doing nothing does not count as success."""
    return random_actions(task, seed, episode, max_steps=max_steps, scale=0.0, size=size)


def fake(task: str, seed: int, episode: int, mode: str = "ok", sleep: float = 0.0) -> dict:
    """No simulator. For testing the runner: mode = ok | crash | hang."""
    time.sleep(sleep)
    if mode == "crash":
        raise RuntimeError("fake crash")
    if mode == "hang":
        time.sleep(3600)
    return {"success": seed % 2 == 0, "llm_calls": 2, "input_tokens": 100,
            "output_tokens": 10, "cost_usd": 0.001}
