"""points_3d: the training point-map image (stored depth) equals the eval one (live depth) for the same frame."""
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
DEPTH = Path(os.environ.get("RUA_DEPTH", str(Path.home() / "apollo-lab/runs/depth")))


@pytest.mark.sim
def test_pointmap_train_eval_match():
    from policy import paths, sim
    from policy.rerender import PointMapStore
    from policy.variants import pointmap_to_image
    task, ep, t = "PickPlaceCounterToCabinet", 2, 50
    store = PointMapStore(DEPTH)
    if not store.has(task, ep):
        pytest.skip("no re-rendered depth for this demo")
    train_img = pointmap_to_image(store.point_map(task, ep, t, cam="left", frame="base"))
    # eval side: live env at the same saved state, through PolicyRunner._images' code path
    from policy.eval import PolicyRunner
    env = sim.make_env(paths.task_dir(task), cams=["left", "right"], depth=True)
    states, _, _ = sim.episode_extras(paths.task_dir(task), ep)
    sim.reset_to_episode(env, paths.task_dir(task), ep)
    env.sim.set_state_from_flattened(states[t]); env.sim.forward()
    obs = env._get_observations(force_update=True)
    r = PolicyRunner.__new__(PolicyRunner)
    r.format, r.cams, r.env, r.cmd = "points_3d", ["left"], env, None
    eval_img = r._images(obs)[0]
    env.close()
    diff = np.abs(train_img.astype(int) - eval_img.astype(int))
    assert diff.mean() < 1.0, diff.mean()
    assert np.percentile(diff, 99) <= 3
