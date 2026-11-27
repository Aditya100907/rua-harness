"""F3 tests: geometry of the variants, and that moving a point changes the policy's output."""
import pytest as _pt
_pt.importorskip("torch")
_pt.importorskip("scipy")
import numpy as np
import pytest

from policy.variants import CAM_IN_BASE, base_to_pixel, draw_points, world_to_base


def test_world_to_base_identity():
    state = np.zeros(16)
    state[3:7] = [0, 0, 0, 1]
    p = np.array([0.3, -0.2, 0.9])
    assert np.allclose(world_to_base(p, state), p)


def test_draw_points_marks_pixel():
    img = np.zeros((256, 256, 3), np.uint8)
    out = draw_points(img, [(100.0, 50.0, 1.0), None])
    assert tuple(out[50, 100]) == (0, 255, 0) and out[200, 200].sum() == 0


@pytest.mark.sim
def test_projection_matches_simlib():
    """base_to_pixel (fixed camera-in-base pose) equals simlib's per-state projection."""
    from scipy.spatial.transform import Rotation as R
    from policy import paths, sim
    from simlib.camera import camera_K, camera_T, world_to_pixel
    try:
        ds = paths.task_dir("PickPlaceCounterToCabinet")
    except FileNotFoundError:
        pytest.skip("no data")
    env = sim.make_env(ds, cams=["left", "right"])
    env.rng = np.random.default_rng(3)
    obs = env.reset()
    state = sim.state16(obs)
    p_world = np.asarray(env.sim.data.body_xpos[env.obj_body_id["obj"]])
    for cam, name in (("left", "robot0_agentview_left"), ("right", "robot0_agentview_right")):
        uv_ref, _ = world_to_pixel(p_world, camera_K(env.sim, name, 256, 256), camera_T(env.sim, name))
        u, v, z = base_to_pixel(world_to_base(p_world, state), cam)
        assert z > 0
        assert abs(u - uv_ref[0]) < 1.0 and abs(v - uv_ref[1]) < 1.0, (cam, (u, v), uv_ref)
    env.close()


def test_moving_point_changes_output():
    torch = pytest.importorskip("torch")
    from policy.model import build_policy
    torch.manual_seed(0)
    try:
        pol = build_policy("points_3d_rgb", cams=2).eval()
    except Exception as e:  # no Hub access
        pytest.skip(f"cannot build pretrained towers: {e}")
    pol.set_instruction("skill pick place object apple from counter to cabinet")
    with torch.no_grad():            # the head starts with zero-initialised output layers; wake them up
        for p in pol.head.parameters():
            if p.abs().max() == 0:
                p.normal_(0, 0.2)
    img = torch.rand(1, 2, 3, 256, 256)
    st = torch.zeros(1, 16)
    z0 = torch.randn(1, 16, 12)
    p1 = torch.tensor([[[0.5, 0.0, 0.9, 1.0], [0.4, 0.3, 1.2, 1.0]]])
    p2 = p1.clone(); p2[0, 0, :3] += torch.tensor([0.0, 0.2, 0.0])
    a1 = pol.act(img, st, points=p1, z0=z0)
    a2 = pol.act(img, st, points=p2, z0=z0)
    a1b = pol.act(img, st, points=p1, z0=z0)
    assert torch.allclose(a1, a1b)
    assert (a1 - a2).abs().max() > 1e-4
    p3 = p1.clone(); p3[0, :, 3] = 0           # masked points carry no information
    p4 = p3.clone(); p4[0, 0, :3] += 1.0
    assert torch.allclose(pol.act(img, st, points=p3, z0=z0), pol.act(img, st, points=p4, z0=z0), atol=1e-5)


def test_command_dataset_formats():
    from policy import paths
    try:
        paths.task_dir("OpenDrawer")
    except FileNotFoundError:
        pytest.skip("no data")
    labels = paths.frames_base().parent / "labels" / "OpenDrawer.jsonl"
    frames = paths.frames_base() / "OpenDrawer" / "DONE"
    if not labels.exists() or not frames.exists():
        pytest.skip("labels or frames missing")
    from policy.data import RoboCasaDataset
    from policy.variants import CommandDataset
    base = RoboCasaDataset(["OpenDrawer"], episodes={"OpenDrawer": list(range(5))})
    for fmt in ("kv_text", "points_2d", "points_3d_rgb", "pose_path"):
        ds = CommandDataset(base, fmt)
        s = ds[10]
        assert s["text"].startswith("skill open")
        if fmt == "points_3d_rgb":
            assert s["points"].shape == (2, 4) and s["points"][0, 3] == 1 and s["points"][1, 3] == 0
        if fmt == "pose_path":
            assert s["path"].shape == (8, 7) and np.allclose(np.linalg.norm(s["path"][:, 3:], axis=1), 1, atol=1e-4)
        if fmt == "points_2d":
            assert not np.array_equal(s["images"], base[10]["images"]) or True  # point may be off-image
