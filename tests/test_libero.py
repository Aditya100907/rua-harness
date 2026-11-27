"""LIBERO data loader on a tiny synthetic hdf5 (no simulator, no download)."""
import json

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("cv2")


def _fake_suite(root, suite="libero_fake", name="put_the_bowl_on_the_plate", n_demos=3, T=20):
    d = root / suite
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    with h5py.File(d / f"{name}_demo.hdf5", "w") as f:
        g = f.create_group("data")
        g.attrs["problem_info"] = json.dumps({"language_instruction": "put the bowl on the plate"})
        for i in range(n_demos):
            e = g.create_group(f"demo_{i}")
            e["actions"] = rng.uniform(-1, 1, (T + i, 7))
            e["states"] = rng.normal(size=(T + i, 40))
            img = np.zeros((T + i, 128, 128, 3), np.uint8)
            img[:, :10] = 255                       # bright rows at the top of the stored (upside-down) image
            e["obs/agentview_rgb"] = img
            e["obs/eye_in_hand_rgb"] = img
            e["obs/ee_states"] = rng.normal(size=(T + i, 6))
            e["obs/gripper_states"] = rng.normal(size=(T + i, 2))
    return f"{suite}/{name}"


def test_libero_dataset(tmp_path):
    from policy.libero.data import LiberoDataset, suite_tasks
    key = _fake_suite(tmp_path)
    assert suite_tasks("libero_fake", tmp_path) == [key]
    ds = LiberoDataset([key], root=tmp_path)
    assert len(ds) == 20 + 21 + 22
    s = ds[len(ds) - 1]
    assert s["images"].shape == (2, 256, 256, 3) and s["images"].dtype == np.uint8
    assert s["state"].shape == (8,) and s["actions"].shape == (16, 7)
    assert s["mask"].sum() == 1 and s["text"] == "put the bowl on the plate"
    # flipped upright: the bright stored-top rows end up at the bottom
    assert s["images"][0, -5:].mean() > 200 and s["images"][0, :5].mean() < 5
    # stage-1 normalization round-trips
    qs, qa = ds.quant[0]
    np.testing.assert_allclose(qa.unnormalize(s["actions"][0]), ds.raw(len(ds) - 1)["actions"][0], atol=1e-5)


def test_quat2axisangle():
    from scipy.spatial.transform import Rotation as R
    from policy.libero.sim import quat2axisangle
    rv = np.array([0.3, -0.2, 0.5])
    np.testing.assert_allclose(quat2axisangle(R.from_rotvec(rv).as_quat()), rv, atol=1e-9)
    assert np.allclose(quat2axisangle(np.array([0, 0, 0, 1.0])), 0)
