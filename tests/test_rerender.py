"""Depth store round trip, without the simulator."""
import numpy as np

from policy import rerender as RR
from simlib import camera as C


def test_group_codec_is_exact():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 65536, size=(5, 16, 16), dtype=np.uint16)
    a[2] = 0
    a[3] = 65535
    assert np.array_equal(RR.decode_group(RR.encode_group(a), 5, 16, 16), a)


def test_depth_to_mm_clips_invalid():
    mm = RR.depth_to_mm(np.array([[0.0, 1.2344, 70.0, np.inf]]))
    assert mm.tolist() == [[0, 1234, 0, 0]]


def test_store_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    T, H = 11, 32                                   # 11 frames: groups of 8 and 3
    depth = rng.uniform(0.5, 3.0, size=(T, H, H))
    K = C.intrinsics(60.0, H, H)
    Tc = np.tile(np.eye(4), (T, 1, 1))
    Tc[:, :3, 3] = rng.normal(size=(T, 3))
    Tb = np.tile(np.eye(4), (T, 1, 1))
    Tb[:, :3, 3] = rng.normal(size=(T, 3))
    mm = RR.depth_to_mm(depth)
    blobs = [RR.encode_group(mm[s:s + RR.GROUP]) for s in range(0, T, RR.GROUP)]
    off = np.r_[0, np.cumsum([len(b) for b in blobs])]
    arrays = {"cams": np.array(["left"]), "size": np.int64(H), "n_frames": np.int64(T),
              "group": np.int64(RR.GROUP), "K_left": K, "T_world_cam_left": Tc.astype(np.float32),
              "T_world_base": Tb.astype(np.float32), "off_left": off,
              "dz_left": np.frombuffer(b"".join(blobs), dtype=np.uint8)}
    RR.save_episode(RR.episode_file(tmp_path, "Fake", 3), arrays)
    store = RR.PointMapStore(tmp_path)
    for t in (10, 0, 7, 8):
        assert np.abs(store.depth("Fake", 3, t, "left") - depth[t]).max() <= 0.0005 + 1e-6
        pm = store.point_map("Fake", 3, t, "left", "world")
        ref = C.depth_to_points(mm[t] / 1000.0, K, Tc[t].astype(np.float32).astype(np.float64))
        assert pm.shape == (3, H, H)
        assert np.abs(pm.transpose(1, 2, 0) - ref).max() < 1e-4
        pb = store.point_map("Fake", 3, t, "left", "base")
        back = np.einsum("ij,jhw->ihw", Tb[t][:3, :3], pb) + Tb[t][:3, 3][:, None, None]
        assert np.abs(back - pm).max() < 1e-4
