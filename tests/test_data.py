"""Data loader tests on real OpenDrawer demos. Skipped when the dataset is not on this machine."""
import numpy as np
import pytest

from policy import paths

try:
    DS = paths.task_dir("OpenDrawer")
    HAVE = (DS / "videos").exists()
except FileNotFoundError:
    HAVE = False
pytestmark = pytest.mark.skipif(not HAVE, reason="OpenDrawer demos not on this machine")

N_EP = 3


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    from policy.data import RoboCasaDataset
    from policy.extract_frames import extract_task
    fb = tmp_path_factory.mktemp("frames")
    extract_task("OpenDrawer", ["left", "right"], fb, max_episodes=N_EP)
    return RoboCasaDataset(["OpenDrawer"], cams=("left", "right"), frames_base=fb,
                           episodes={"OpenDrawer": list(range(N_EP))})


def test_shapes_and_dtypes(ds):
    s = ds[5]
    assert s["images"].shape == (2, 256, 256, 3) and s["images"].dtype == np.uint8
    assert s["state"].shape == (16,) and s["state"].dtype == np.float32
    assert s["actions"].shape == (16, 12) and s["actions"].dtype == np.float32
    assert s["mask"].shape == (16,) and s["mask"].all()
    assert s["text"] in ("Open the right drawer.", "Open the left drawer.")


def test_value_ranges(ds):
    xs = np.stack([ds[i]["state"] for i in range(0, len(ds), 50)])
    acts = np.stack([ds[i]["actions"] for i in range(0, len(ds), 50)])
    assert np.isfinite(xs).all() and np.isfinite(acts).all()
    assert np.abs(xs).max() < 5 and np.abs(acts).max() < 5      # quantile-scaled, outliers can pass +-1
    raw = np.stack([ds.raw(i)["actions"] for i in range(0, len(ds), 50)])
    assert set(np.unique(raw[..., 11])) <= {-1.0, 1.0}             # gripper
    assert np.allclose(raw[..., 4], -1.0)                           # control mode: arm


def test_camera_order(ds):
    """Frame store images equal a direct decode of the left then right video."""
    from policy.extract_frames import decode_video, video_path
    i = 7
    s = ds[i]
    ep, t = s["episode"], s["frame"]
    left = decode_video(video_path(DS, "left", ep))[t]
    right = decode_video(video_path(DS, "right", ep))[t]
    d_l = np.abs(s["images"][0].astype(int) - left).mean()
    d_r = np.abs(s["images"][1].astype(int) - right).mean()
    d_swap = np.abs(s["images"][0].astype(int) - right).mean()
    assert d_l < 3 and d_r < 3, (d_l, d_r)      # JPEG q95 error is about 1 grey level
    assert d_swap > 2 * d_l


def test_chunks_do_not_cross_episodes(ds):
    n0 = int(ds.tables[0]["lens"][0])
    last = n0 - 1
    i = int(np.where((ds.index[:, 1] == 0) & (ds.index[:, 2] == last))[0][0])
    s = ds[i]
    assert s["mask"].sum() == 1 and s["mask"][0]
    assert np.allclose(s["actions"], s["actions"][0])              # padded with the episode's last action
    raw_last = ds.tables[0]["actions"][n0 - 1]
    assert np.allclose(ds.raw(i)["actions"][-1], raw_last)
    j = i + 1                                                       # first frame of episode 1
    assert ds.index[j, 1] == 1 and ds.index[j, 2] == 0


def test_normalize_roundtrip(ds):
    from policy.data import task_quantiles
    qs, qa = task_quantiles(ds.norm_stats, "OpenDrawer")
    raw = ds.raw(11)
    assert np.allclose(qa.unnormalize(qa.normalize(raw["actions"])), raw["actions"], atol=1e-9)
    assert np.allclose(qs.unnormalize(qs.normalize(raw["state"])), raw["state"], atol=1e-9)


def test_roundtrip_matches_dem_normalizer(ds):
    torch = pytest.importorskip("torch")
    from dem import Normalizer
    t = ds.norm_stats["per_task"]["OpenDrawer"]["action"]
    n = Normalizer.from_quantiles(torch.tensor(t["q01"]), torch.tensor(t["q99"]))
    from policy.data import task_quantiles
    _, qa = task_quantiles(ds.norm_stats, "OpenDrawer")
    x = ds.raw(3)["actions"]
    assert np.allclose(n.normalize(torch.tensor(x, dtype=torch.float32)).numpy(), qa.normalize(x), atol=1e-4)
    y = np.random.default_rng(0).normal(size=x.shape)
    assert np.allclose(n.unnormalize(torch.tensor(y, dtype=torch.float32)).numpy(), qa.unnormalize(y), atol=1e-4)
