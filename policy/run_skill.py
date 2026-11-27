"""run_skill: run the trained policy on a live RoboCasa env until it stops (E4).

    status = run_skill(env, "Open the left drawer.", max_seconds=30, checkpoint=CKPT)

Stops on the first of: the simulator's success check (see the note below),
`max_seconds` of robot time (20 control steps per second), or `max_steps`.
Returns a status dict and the final camera images (dataset orientation).

Load order: call `_runner(checkpoint, device)` before building the env. On the Mac, loading the
model after the env has been hard-reset twice corrupts every later render (Nov 27). The result
carries `img_rowdiff` (mean change between neighbouring pixel rows: about 6 clean, 25+ corrupt).

Note on stopping: the policy has no learned "done" output yet, so by default the
skill stops when `env._check_success()` turns true. That is simulator ground
truth: it is used here only to stop the skill, and the returned dict keeps it
under `grading_only`, which the agent-facing tool must not show the LLM. Pass
`stop_on_success=False` for a skill that runs to its time limit. A learned done
output is in the backlog (docs/OVERNIGHT.md, Part L).
"""
from __future__ import annotations

import time
from functools import lru_cache

import numpy as np

from policy import sim

CONTROL_HZ = 20


@lru_cache(maxsize=4)
def _runner(checkpoint: str, device: str):
    from policy.eval import PolicyRunner
    return PolicyRunner(checkpoint, device=device)


def default_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def run_skill(env, instruction: str, max_seconds: float = 30.0, checkpoint: str = "", task: str | None = None,
              command: dict | None = None, max_steps: int | None = None, execute: int = 8,
              stop_on_success: bool = True, device: str | None = None, obs: dict | None = None) -> dict:
    """Run one skill. `task` picks the normalization stats (defaults to the env's class name).

    `command`: for formats other than `sentence`, a dict with `kv_text`, `grasp_world`,
    `place_world` (world-frame points or None) and `has` (two booleans), e.g. from the agent's
    lifted pixels. For `sentence` checkpoints only `instruction` is used.
    """
    t0 = time.time()
    runner = _runner(checkpoint, device or default_device())
    task = task or type(env).__name__
    if runner.format == "sentence":
        runner.set_task(task, instruction)
    else:
        from policy.data import task_quantiles
        if command is None:
            raise ValueError(f"checkpoint format {runner.format!r} needs a `command` dict")
        runner.quant = task_quantiles(runner.stats, task)
        runner.cmd = command
        runner.env = env
        runner.policy.set_instruction(command["kv_text"])
    if obs is None:
        obs = env._get_observations(force_update=True)
    limit = int(max_seconds * CONTROL_HZ)
    if max_steps is not None:
        limit = min(limit, max_steps)
    steps, success, reason, model_s = 0, False, "max_seconds", 0.0
    while steps < limit:
        tm = time.time()
        chunk = runner.act(obs)
        model_s += time.time() - tm
        for a in sim.lerobot_to_env_action(chunk[:execute]):
            obs, _, _, _ = env.step(a)
            steps += 1
            if env._check_success():
                success = True
                if stop_on_success:
                    break
            if steps >= limit:
                break
        if success and stop_on_success:
            reason = "success_check"
            break
    if max_steps is not None and steps >= max_steps and reason != "success_check":
        reason = "max_steps"
    imgs = {c: sim.images(obs, [c])[0] for c in ("left", "right", "wrist")
            if f"{sim.CAMERAS[c]}_image" in obs}
    rowdiff = float(np.abs(np.diff(sim.images(obs, ["left"])[0].astype(np.int16), axis=0)).mean())
    return {"status": "stopped", "reason": reason, "img_rowdiff": round(rowdiff, 1), "robot_seconds": steps / CONTROL_HZ, "steps": steps,
            "wall_seconds": round(time.time() - t0, 2), "model_seconds": round(model_s, 2),
            "images": imgs, "obs": obs, "grading_only": {"success": bool(success)}}
