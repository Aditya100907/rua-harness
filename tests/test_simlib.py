"""Fast simlib tests: no simulator needed."""
import csv

import numpy as np
import pytest

from simlib import camera as C
from simlib.runner import COLUMNS, make_row, run_episodes
from simlib.tasks import ATOMIC_SEEN, HORIZONS
from simlib.truth import TASK_TARGETS


def random_pose(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return C.extrinsics_from_mujoco(rng.normal(size=3), q)


def test_intrinsics_formula():
    K = C.intrinsics(90.0, 256, 256)
    assert np.allclose(K, [[128, 0, 128], [0, 128, 128], [0, 0, 1]])


def test_metric_depth_endpoints():
    assert np.isclose(C.metric_depth(np.array([[0.0]]), 0.1, 50.0)[0, 0], 0.1)
    assert np.isclose(C.metric_depth(np.array([[1.0]]), 0.1, 50.0)[0, 0], 50.0)
    assert C.metric_depth(np.zeros((4, 5, 1)), 0.1, 50.0).shape == (4, 5)


def test_flip_rows_is_self_inverse():
    a = np.arange(12).reshape(4, 3)
    assert np.array_equal(C.flip_rows(C.flip_rows(a)), a)
    assert np.array_equal(C.flip_rows(a)[0], a[-1])


def test_camera_axes_and_row_convention():
    # MuJoCo camera with identity orientation looks down world -z, world +y is image-up.
    T = C.extrinsics_from_mujoco(np.zeros(3), np.eye(3))
    K = C.intrinsics(60.0, 100, 100)
    uv, z = C.world_to_pixel([0.0, 0.0, -2.0], K, T)
    assert np.isclose(z, 2.0) and np.allclose(uv, [49.5, 49.5])
    uv_up, _ = C.world_to_pixel([0.0, 0.5, -2.0], K, T)      # higher in the world
    uv_right, _ = C.world_to_pixel([0.5, 0.0, -2.0], K, T)
    assert uv_up[1] < 49.5            # appears in a smaller row index: top-row-first
    assert uv_right[0] > 49.5


def test_pixel_world_round_trip():
    rng = np.random.default_rng(0)
    K = C.intrinsics(45.0, 256, 256)
    for _ in range(5):
        T = random_pose(rng)
        uv = rng.integers(0, 256, size=(50, 2))
        z = rng.uniform(0.3, 3.0, size=50)
        p = C.pixel_to_world(uv, z, K, T)
        uv2, z2 = C.world_to_pixel(p, K, T)
        assert np.allclose(uv2, uv) and np.allclose(z2, z)


def test_depth_image_lookup_and_point_map():
    rng = np.random.default_rng(1)
    K, T = C.intrinsics(45.0, 32, 48), random_pose(rng)
    depth = rng.uniform(0.5, 2.0, size=(32, 48))
    pm = C.depth_to_points(depth, K, T)
    assert pm.shape == (32, 48, 3)
    assert np.allclose(pm[5, 7], C.pixel_to_world([7, 5], depth, K, T))   # uv = (col, row)


def test_task_tables_cover_atomic_seen():
    assert len(ATOMIC_SEEN) == 18 == len(set(ATOMIC_SEEN))
    assert set(TASK_TARGETS) == set(ATOMIC_SEEN) == set(HORIZONS)


def test_copies_match_robocasa_registry():
    reg = pytest.importorskip("robocasa.utils.dataset_registry")
    assert list(ATOMIC_SEEN) == reg.TARGET_TASKS["atomic_seen"]
    assert all(reg.ATOMIC_TASK_DATASETS[t]["horizon"] == HORIZONS[t] for t in ATOMIC_SEEN)


def test_make_row_has_schema_columns():
    row = make_row("OpenDrawer", 3, 7, {"success": True}, 1.234, "abc")
    assert list(row) == COLUMNS
    assert row["success"] is True and row["llm_calls"] == 0 and row["seconds"] == 1.23
    assert make_row("OpenDrawer", 0, 0, None, 5, "abc")["success"] is False


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def test_runner_parallel_rows_and_logs(tmp_path):
    out = tmp_path / "run.csv"
    run_episodes("simlib.episodes:fake", ["A", "B"], 2, out, seed0=10, workers=4, timeout=60)
    rows = read(out)
    assert len(rows) == 4 and list(rows[0]) == COLUMNS
    for r in rows:
        assert r["success"] == str(int(r["seed"]) % 2 == 0)
        assert r["llm_calls"] == "2" and r["code_commit"]
    assert len(list((tmp_path / "run_logs").glob("*.log"))) == 4


def test_runner_crash_and_timeout(tmp_path):
    out = tmp_path / "bad.csv"
    run_episodes("simlib.episodes:fake", ["Crash"], 1, out, kwargs={"mode": "crash"}, timeout=60)
    run_episodes("simlib.episodes:fake", ["Hang"], 1, out, kwargs={"mode": "hang"}, timeout=3)
    rows = read(out)
    assert [r["success"] for r in rows] == ["False", "False"]
    assert float(rows[1]["seconds"]) < 30
    logs = {p.name: p.read_text() for p in (tmp_path / "bad_logs").glob("*.log")}
    assert "fake crash" in logs["Crash_ep000.log"] and "crashed" in logs["Crash_ep000.log"]
    assert "timeout" in logs["Hang_ep000.log"]
