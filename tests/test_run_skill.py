"""E4: run_skill runs a checkpoint on a live env and returns a status dict and images."""
import os
from pathlib import Path

import pytest

CKPT = os.environ.get("RUA_TEST_CKPT", str(Path.home() / "apollo-lab/runs/train_test/ckpt_0000050.pt"))


@pytest.mark.sim
def test_run_skill_live_opendrawer():
    if not Path(CKPT).exists():
        pytest.skip("no test checkpoint")
    import numpy as np
    from policy import paths, sim
    from policy.run_skill import run_skill
    env = sim.make_env(paths.task_dir("OpenDrawer"), cams=["left", "right"])
    env.rng = np.random.default_rng(5)
    obs = env.reset()
    out = run_skill(env, env.get_ep_meta()["lang"], max_seconds=2.0, checkpoint=CKPT, task="OpenDrawer", obs=obs)
    env.close()
    assert out["status"] == "stopped" and out["reason"] in ("max_seconds", "success_check")
    assert out["steps"] == 40 or out["grading_only"]["success"]
    assert out["images"]["left"].shape == (256, 256, 3)
    assert "success" not in {k for k in out if k != "grading_only"}
