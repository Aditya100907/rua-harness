"""Command-format variants (F3): dataset wrapper and the point-token policy.

Geometry: label points are stored in the world frame (policy/label_commands.py).
At frame t they are moved into the robot base frame of frame t, using the base
pose in the state vector (slots 0:7, quaternion xyzw). The two agent-view
cameras are fixed on the mobile base; their pose in the base frame was measured
in the simulator (identical across scenes and over a demo):
  left : t = (-0.5,  0.35, 1.05), q_xyzw = (-0.5562,  0.6775, -0.3768,  0.2994)
  right: t = (-0.5, -0.35, 1.05), q_xyzw = ( 0.6775, -0.5562,  0.2994, -0.3768)
  intrinsics: f = 221.703 px, c = (128, 128) at 256 x 256 (fovy 60 deg), OpenCV axes,
  image rows top-first (dataset orientation).

Point tokens: each point (grasp, place) goes through Fourier features and a small
MLP to one token of the language width, plus a learned slot embedding, and the
tokens are appended to the language tokens (mask False when a point is absent).
The DEM head itself is unchanged; it sees 32 + 2 language-stream tokens.
Pose path: 8 future gripper poses (position + quaternion in the current base
frame, every 8 frames = 0.4 s) become 8 such tokens.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R

CAM_IN_BASE = {
    "left": ((-0.5, 0.35, 1.05), (-0.5562, 0.6775, -0.3768, 0.2994)),
    "right": ((-0.5, -0.35, 1.05), (0.6775, -0.5562, 0.2994, -0.3768)),
}
K256 = np.array([[221.703, 0, 128.0], [0, 221.703, 128.0], [0, 0, 1]])
PATH_LEN, PATH_STRIDE = 8, 8
POINT_FORMATS = ("points_3d", "points_3d_rgb")
POINTMAP_FORMATS = ("points_3d", "points_3d_dist")   # color replaced by a geometric image (needs depth)
DIST_FORMATS = ("points_3d_dist",)   # per-pixel distance to the grasp and place points + height (no point tokens)
PM_RANGE = 1.5                              # metres: base-frame xyz in [-1.5, 1.5] -> [0, 1]


def distmap_to_image(pm: np.ndarray, grasp, place) -> np.ndarray:
    """[3, H, W] base-frame xyz + base-frame command points -> uint8 [H, W, 3]: distance to the grasp
    point, distance to the place point (0-1.5 m -> 255-0, nearer is brighter; absent point -> 0), and height
    (z in [-1.5, 1.5] m). The command enters through the vision path instead of as appended tokens."""
    xyz = np.nan_to_num(pm.transpose(1, 2, 0), nan=1e3)
    chans = []
    for p in (grasp, place):
        if p is None:
            chans.append(np.zeros(xyz.shape[:2]))
        else:
            d = np.linalg.norm(xyz - np.asarray(p)[None, None], axis=-1)
            chans.append(np.clip(1 - d / 1.5, 0, 1))
    chans.append(np.clip((np.nan_to_num(pm[2], nan=-PM_RANGE) + PM_RANGE) / (2 * PM_RANGE), 0, 1))
    return (np.stack(chans, -1) * 255).round().astype(np.uint8)


def pointmap_to_image(pm: np.ndarray) -> np.ndarray:
    """[3, H, W] base-frame xyz (NaN = invalid) -> uint8 [H, W, 3] image for the vision encoder."""
    x = np.nan_to_num((pm + PM_RANGE) / (2 * PM_RANGE), nan=0.0)
    return (np.clip(x, 0, 1) * 255).round().astype(np.uint8).transpose(1, 2, 0)
TEXT_FORMATS = ("kv_text", "points_2d", "points_3d", "points_3d_rgb", "pose_path")


def world_to_base(p_world: np.ndarray, state: np.ndarray) -> np.ndarray:
    return R.from_quat(state[3:7]).inv().apply(np.asarray(p_world) - state[0:3])


def base_to_pixel(p_base: np.ndarray, cam: str) -> tuple[float, float, float]:
    """(u, v, depth) of a base-frame point in a 256 x 256 agent-view image."""
    t, q = CAM_IN_BASE[cam]
    pc = R.from_quat(q).inv().apply(np.asarray(p_base) - np.asarray(t))
    uvw = K256 @ pc
    # pixel-index coordinates with the half-pixel shift of simlib.camera.world_to_pixel
    return uvw[0] / uvw[2] - 0.5, uvw[1] / uvw[2] - 0.5, pc[2]


def draw_points(img: np.ndarray, pts: list, colors=((0, 255, 0), (255, 0, 0)), r: int = 4) -> np.ndarray:
    """Draw marks at (u, v) pixels; pts entries may be None. r <= 5: the original small dot (black outline +
    colored disk). r > 5: a high-contrast mark (black disk, white ring, colored centre), PROPOSALS P-2."""
    import cv2
    out = np.ascontiguousarray(img.copy())
    for p, c in zip(pts, colors):
        if p is None:
            continue
        u, v, z = p
        if z <= 0 or not (0 <= u < out.shape[1] and 0 <= v < out.shape[0]):
            continue
        uv = (int(round(u)), int(round(v)))
        if r <= 5:
            cv2.circle(out, uv, r + 1, (0, 0, 0), -1)
            cv2.circle(out, uv, r, c, -1)
        else:
            cv2.circle(out, uv, r + 2, (0, 0, 0), -1)
            cv2.circle(out, uv, r, (255, 255, 255), -1)
            cv2.circle(out, uv, max(2, r - 4), c, -1)
    return out


def base_to_world(p_base: np.ndarray, state: np.ndarray) -> np.ndarray:
    return R.from_quat(state[3:7]).apply(np.asarray(p_base)) + state[0:3]


def eef_pose_base(state: np.ndarray) -> np.ndarray:
    """[..., 7] end-effector position + quaternion in the base frame (straight from the state)."""
    return state[..., 7:14]


class CommandDataset:
    """Wraps RoboCasaDataset: replaces the text with the format's command text and adds geometry."""

    def __init__(self, base, fmt: str, labels_dir=None, point_noise_cm: float = 2.0, rot_noise_deg: float = 3.0,
                 name_dropout: float = 0.0, mark_radius: int = 4, mark_all_cams: bool = False,
                 noise_per_episode: bool = False):
        from pathlib import Path
        from policy import paths
        from policy.label_commands import load_labels
        self.base, self.fmt = base, fmt
        self.noise_m, self.rot_noise = point_noise_cm / 100.0, rot_noise_deg
        # name dropout: with this probability the object name in the command text becomes "object", so the
        # policy has to find the object from the point (same token as policy.commands_gt drop="name")
        self.name_dropout = name_dropout
        # PROPOSALS P-2: mark size, marks on every camera (incl. the wrist, projected with the per-frame camera
        # poses saved with the re-rendered depth), and point noise held fixed for the whole episode
        self.mark_radius, self.mark_all_cams, self.noise_per_episode = mark_radius, mark_all_cams, noise_per_episode
        ld = Path(labels_dir) if labels_dir else paths.frames_base().parent / "labels"
        self.labels = [{lb["episode"]: lb for lb in load_labels(t, ld)} for t in base.tasks]
        self.norm_stats, self.tables, self.tasks = base.norm_stats, base.tables, base.tasks
        self.rng = None          # a fresh generator per sample, so DataLoader workers do not share noise
        self._pm = None

    def __len__(self):
        return len(self.base)

    def instructions(self):
        if self.fmt == "sentence":
            return self.base.instructions()
        texts = {lb["kv_text"] for d in self.labels for lb in d.values()}
        if self.name_dropout:
            texts |= {dropped_name_text(lb) for d in self.labels for lb in d.values()}
        return sorted(texts)

    def __getitem__(self, i: int) -> dict:
        s = self.base[i]
        ti, ep, t = s["task"], s["episode"], s["frame"]
        lb = self.labels[ti][ep]
        if self.fmt != "sentence":
            s["text"] = lb["kv_text"]
            if self.name_dropout and np.random.default_rng().random() < self.name_dropout:
                s["text"] = dropped_name_text(lb)
        tb = self.tables[ti]
        g = int(tb["starts"][ep]) + t
        state = tb["states"][g]
        rng = np.random.default_rng() if self.rng is None else self.rng
        if self.noise_per_episode:      # same noise draw for every frame of this episode
            rng = np.random.default_rng(int(ti) * 1_000_003 + int(ep))
        pts, mask = [], []
        for key in ("grasp_world", "place_world"):
            p = lb.get(key)
            if p is None:
                pts.append(np.zeros(3)); mask.append(False)
            else:
                pb = world_to_base(np.asarray(p), state) + rng.normal(0, self.noise_m, 3)
                pts.append(pb); mask.append(True)
        if self.fmt == "points_2d":
            imgs = []
            for k, cam in enumerate(self.base.cams):
                if self.mark_all_cams:
                    from simlib.camera import world_to_pixel
                    if self._pm is None:
                        from policy.rerender import PointMapStore
                        self._pm = PointMapStore()
                    K, T = self._pm.camera(self.tasks[ti], ep, t, cam, frame="world")
                    uv = []
                    for p, m in zip(pts, mask):
                        if not m:
                            uv.append(None)
                            continue
                        (u, v), z = world_to_pixel(base_to_world(p, state), K, T)
                        uv.append((u, v, z))
                    imgs.append(draw_points(s["images"][k], uv, r=self.mark_radius))
                elif cam in CAM_IN_BASE:
                    uv = [base_to_pixel(p, cam) if m else None for p, m in zip(pts, mask)]
                    imgs.append(draw_points(s["images"][k], uv, r=self.mark_radius))
                else:
                    imgs.append(s["images"][k])
            s["images"] = np.stack(imgs)
        if self.fmt in POINTMAP_FORMATS:
            if self._pm is None:
                from policy.rerender import PointMapStore
                self._pm = PointMapStore()
            pms = [self._pm.point_map(self.tasks[ti], ep, t, cam=c, frame="base") for c in self.base.cams]
            if self.fmt in DIST_FORMATS:
                gp = pts[0] if mask[0] else None
                pp = pts[1] if mask[1] else None
                s["images"] = np.stack([distmap_to_image(pm, gp, pp) for pm in pms])
            else:
                s["images"] = np.stack([pointmap_to_image(pm) for pm in pms])
        if self.fmt in POINT_FORMATS:
            s["points"] = np.concatenate([np.stack(pts).astype(np.float32),
                                          np.asarray(mask, np.float32)[:, None]], 1)   # [2, 4]
        if self.fmt == "pose_path":
            n = int(tb["lens"][ep])
            idx = int(tb["starts"][ep]) + np.minimum(t + PATH_STRIDE * np.arange(1, PATH_LEN + 1), n - 1)
            fut = tb["states"][idx]                                   # future states
            # future eef poses, expressed in the current base frame
            p_w = fut[:, 0:3] + R.from_quat(fut[:, 3:7]).apply(fut[:, 7:10])
            p_b = R.from_quat(state[3:7]).inv().apply(p_w - state[0:3]) + rng.normal(0, self.noise_m, (PATH_LEN, 3))
            q_w = R.from_quat(fut[:, 3:7]) * R.from_quat(fut[:, 10:14])
            q_b = R.from_quat(state[3:7]).inv() * q_w
            q_b = R.from_rotvec(rng.normal(0, np.deg2rad(self.rot_noise), (PATH_LEN, 3))) * q_b
            s["path"] = np.concatenate([p_b, q_b.as_quat()], 1).astype(np.float32)   # [8, 7]
        return s


def dropped_name_text(lb: dict) -> str:
    """The label's command text with the object name replaced by "object" (as commands_gt drop="name")."""
    from rua_harness.command import Command, Ref, to_policy_text
    c = lb["command"]
    cmd = Command(skill=c["skill"], object=Ref("object") if c.get("object") else None,
                  target=Ref(c["target"]["name"]) if c.get("target") else None,
                  source=c.get("source"), direction=c.get("direction"))
    return to_policy_text(cmd)


def collate_variant(batch):
    from policy.data import collate
    out = collate(batch)
    for k in ("points", "path"):
        if k in batch[0]:
            out[k] = torch.as_tensor(np.stack([b[k] for b in batch]))
    return out


class Fourier(nn.Module):
    def __init__(self, dim_in: int, n: int = 16, scale: float = 10.0):
        super().__init__()
        self.register_buffer("B", torch.randn(dim_in, n) * scale, persistent=True)

    def forward(self, x):
        z = 2 * np.pi * x @ self.B
        return torch.cat([x, z.sin(), z.cos()], -1)


class GeoEncoder(nn.Module):
    """[B, N, d_in] -> [B, N, d_out] tokens, plus a learned per-slot embedding."""

    def __init__(self, d_in: int, n_slots: int, d_out: int, hidden: int = 256, n_fourier: int = 16):
        super().__init__()
        self.ff = Fourier(d_in, n_fourier)
        self.mlp = nn.Sequential(nn.Linear(d_in + 2 * n_fourier, hidden), nn.GELU(), nn.Linear(hidden, hidden),
                                 nn.GELU(), nn.Linear(hidden, d_out))
        self.slot = nn.Parameter(torch.randn(n_slots, d_out) * 0.02)

    def forward(self, x):
        return self.mlp(self.ff(x)) + self.slot


def _make_pointdem():
    from dem import DEM

    class PointDEM(DEM):
        """DEM plus geometric tokens appended to the language stream."""

        def __init__(self, config=None, fmt: str = "points_3d_rgb", **kw):
            super().__init__(config, **kw)
            self.rua_format = fmt
            d = self.language.embed_dim
            if fmt in POINT_FORMATS:
                self.geo = GeoEncoder(3, 2, d)
            elif fmt == "pose_path":
                self.geo = GeoEncoder(7, PATH_LEN, d)
            else:
                raise ValueError(fmt)
            self._geo_cmd = None

        def _geo_tokens(self, points=None, path=None):
            if points is not None:
                tok = self.geo(points[..., :3].float())
                mask = points[..., 3] > 0.5
            else:
                tok = self.geo(path.float())
                mask = torch.ones(path.shape[:2], dtype=torch.bool, device=path.device)
            return tok, mask

        def set_geometry(self, points=None, path=None):
            """Store the geometric command used by `act` when none is passed ([N, d] or [1, N, d])."""
            def f(x):
                if x is None:
                    return None
                x = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)
                return x[None] if x.ndim == 2 else x
            self._geo_cmd = (f(points), f(path))

        def _cond(self, vis_tokens, proprio, texts, lang, points, path):
            B = vis_tokens.shape[0]
            if points is None and path is None:
                points, path = self._geo_cmd
            lt, lm = self._language_for(texts, lang, B)
            gt, gm = self._geo_tokens(None if points is None else points.to(self.device),
                                      None if path is None else path.to(self.device))
            if gt.shape[0] == 1 and B > 1:
                gt, gm = gt.expand(B, -1, -1), gm.expand(B, -1)
            lt = torch.cat([lt.to(gt.dtype), gt], 1)
            lm = torch.cat([lm, gm], 1)
            return self.build_cond(vis_tokens, lt, lm, proprio)

        def loss(self, images, proprio, actions, texts=None, lang=None, vis_tokens=None, points=None, path=None):
            if vis_tokens is None:
                vis_tokens = self.encode_vision(images)
            cond = self._cond(vis_tokens, proprio, texts, lang, points, path)
            x = self.action_norm.normalize(actions.to(self.device).float()).to(self.compute_dtype)
            return self.objective.loss(self.head, x, cond)

        @torch.no_grad()
        def act(self, images, proprio, texts=None, lang=None, vis_tokens=None, nfe=1, z0=None, points=None, path=None):
            if vis_tokens is None:
                vis_tokens = self.encode_vision(images)
            if isinstance(texts, str):
                texts = [texts]
            cond = self._cond(vis_tokens, proprio, texts, lang, points, path)
            z = self.objective.sample(self.head, cond, self.config.chunk, self.config.action_dim, nfe=nfe, z0=z0)
            return self.action_norm.unnormalize(z.float())

        def parameter_groups(self, lr_head=1e-4, lr_vision=5e-5):
            g = super().parameter_groups(lr_head, lr_vision)
            g[0]["params"] = g[0]["params"] + list(self.geo.parameters())
            return g

        def save(self, path, include_frozen=False):
            super().save(path, include_frozen)
            ck = torch.load(path, map_location="cpu", weights_only=True)
            ck["rua_format"] = self.rua_format
            torch.save(ck, path)

        @classmethod
        def load(cls, path, map_location="cpu", **overrides):
            from dataclasses import replace
            from dem.policy import DEMConfig, _build_normalizer, _freeze
            ck = torch.load(path, map_location=map_location, weights_only=True)
            cfg = DEMConfig.from_dict(ck["config"], ignore_unknown=True)
            sd = ck["state_dict"]
            has_vision = any(k.startswith("vision.") for k in sd)
            build_cfg = replace(cfg, vision_pretrained=cfg.vision_pretrained and not has_vision)
            model = cls(build_cfg, fmt=ck["rua_format"])
            model.config = cfg
            spec = ck.get("normalizers", {})
            model.action_norm = _build_normalizer(cfg.action_dim, spec.get("action", 1))
            model.proprio_norm = _build_normalizer(cfg.proprio_dim, spec.get("proprio", 1))
            missing, unexpected = model.load_state_dict(sd, strict=False)
            omitted = tuple(ck.get("omitted_prefixes", []))
            bad = [k for k in missing if not k.startswith(omitted)]
            if bad or unexpected:
                raise RuntimeError(f"checkpoint mismatch: missing {bad[:5]}, unexpected {unexpected[:5]}")
            if cfg.freeze_vision:
                _freeze(model.vision)
            return model

    return PointDEM


class _Lazy:
    _cls = None

    def __getattr__(self, name):
        raise AttributeError(name)


def __getattr__(name):           # `from policy.variants import PointDEM` builds the class on first use
    if name == "PointDEM":
        if _Lazy._cls is None:
            _Lazy._cls = _make_pointdem()
        return _Lazy._cls
    raise AttributeError(name)
