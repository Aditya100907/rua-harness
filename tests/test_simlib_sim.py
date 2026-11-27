"""simlib tests that build real RoboCasa envs. Skip with: pytest -m "not sim"."""
import numpy as np
import pytest

from simlib import camera as C
from simlib import truth
from simlib.tasks import ATOMIC_SEEN, check_success, make_env

pytestmark = pytest.mark.sim


@pytest.fixture(scope="module")
def scene():
    env = make_env("PickPlaceCounterToCabinet", seed=0, size=128, depth=True)
    obs = env.reset()
    yield env, obs
    env.close()


def test_views_match_robosuite_and_lift_object(scene):
    import robosuite.utils.camera_utils as CU
    env, obs = scene
    views = C.views(env, obs)
    assert set(views) == set(C.CAMERAS)
    true = truth.object_position(env, "obj")
    geoms = truth.object_geom_ids(env, "obj")
    for cam in ("robot0_agentview_left", "robot0_agentview_right"):
        v = views[cam]
        assert v.rgb.shape == (128, 128, 3) and v.depth.shape == (128, 128)
        assert np.allclose(v.K, CU.get_camera_intrinsic_matrix(env.sim, cam, 128, 128))
        assert np.allclose(v.T_world_cam, CU.get_camera_extrinsic_matrix(env.sim, cam))
        mask = np.isin(C.segmentation(env, cam, 128, 128), geoms)
        assert mask.sum() > 10
        row, col = np.argwhere(mask).mean(0).round().astype(int)
        assert np.linalg.norm(v.pixel_to_world([col, row]) - true) < 0.06
        uv, z = v.world_to_pixel(true)
        u, r = np.round(uv).astype(int)
        assert mask[r - 2:r + 3, u - 2:u + 3].any()


def test_render_view_matches_obs(scene):
    env, obs = scene
    v = C.render_view(env, "robot0_agentview_left", 128, 128)
    o = C.views(env, obs)["robot0_agentview_left"]
    assert np.abs(v.depth - o.depth).mean() < 0.01


@pytest.mark.parametrize("task", ATOMIC_SEEN)
def test_target_part_found(task):
    env = make_env(task, seed=0, size=64, depth=False)
    try:
        env.reset()
        p = truth.target_part(env, task)
        assert np.all(np.isfinite(p.pos)) and p.names
        assert check_success(env) is False   # nothing is solved at reset
    finally:
        env.close()


def test_runner_with_real_env(tmp_path):
    from simlib.runner import run_episodes
    rows = run_episodes("simlib.episodes:random_actions", ["OpenDrawer"], 1, tmp_path / "r.csv",
                        kwargs={"max_steps": 5, "size": 64}, timeout=300)
    assert rows[0]["success"] is False and rows[0]["seconds"] > 0
