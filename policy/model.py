"""Command-format variants as thin wrappers around DEM. One flag switches format.

  sentence       the original instruction, color images (stock DEM)
  kv_text        rua_harness.command.to_policy_text(command), color images
  points_2d      kv_text plus grasp and place points drawn on the color images
  points_3d      kv_text plus the two 3D points as 2 extra context tokens (small MLP), point-map input
  points_3d_rgb  as points_3d but with color images (separates the effect of points and of the image input)
  pose_path      kv_text plus a short future gripper-pose path as extra tokens, point-map input

Only `sentence` needs nothing beyond the demos. The others need hindsight labels
(policy/label_commands.py) and, for point maps, re-rendered depth.
"""
from __future__ import annotations

import torch

FORMATS = ("sentence", "kv_text", "points_2d", "points_3d", "points_3d_rgb", "pose_path")


def build_policy(fmt: str = "sentence", cams: int = 2, offline: bool = False, freeze_vision: bool = False,
                 dispersive_weight: float = 0.25, ratio_r_eq_t: float = 0.5):
    from dem import DEM, DEMConfig
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; choose from {FORMATS}")
    cfg = DEMConfig(action_dim=12, proprio_dim=16, chunk=16, num_cameras=cams, freeze_vision=freeze_vision,
                    vision_pretrained=not offline, language_pretrained=not offline,
                    dispersive_weight=dispersive_weight, ratio_r_eq_t=ratio_r_eq_t)
    if fmt in ("sentence", "kv_text", "points_2d"):
        return DEM(cfg)
    from policy.variants import PointDEM
    return PointDEM(cfg, fmt)


def load_policy(path, offline: bool = False, map_location="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    fmt = ck.get("rua_format", "sentence")
    if fmt in ("sentence", "kv_text", "points_2d"):    # plain DEM weights; the format only changes inputs
        from dem import DEM
        return DEM.load(path, map_location=map_location)
    from policy.variants import PointDEM
    return PointDEM.load(path, map_location=map_location)


def wrap_dataset(ds, args):
    if args.format == "sentence":
        return ds
    from policy.variants import CommandDataset
    return CommandDataset(ds, args.format, labels_dir=args.labels_dir, point_noise_cm=args.point_noise_cm)


class DoneHead(torch.nn.Module):
    """Learned "done" output (Part L backlog): predicts episode progress in [0, 1] from the mean vision
    token and the stage-1 state. Kept outside the DEM module (saved as ckpt_*.done.pt) so DEM checkpoints
    stay loadable by DEM.load. run_skill can stop when progress stays above a threshold."""

    def __init__(self, vis_dim: int = 512, proprio_dim: int = 16, hidden: int = 256):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(vis_dim + proprio_dim, hidden), torch.nn.GELU(),
                                       torch.nn.Linear(hidden, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, 1))

    def forward(self, vis_tokens, proprio):
        return torch.sigmoid(self.net(torch.cat([vis_tokens.float().mean(1), proprio.float()], -1))).squeeze(-1)


def mse_loss(policy, vis, batch, dev, extra):
    """Objective "mse": deterministic regression through the same head. The head sees z = 0, r = 0, t = 1,
    and the action is x = z - u = -u, which is exactly what `act(..., z0=zeros, nfe=1)` returns."""
    x = policy.action_norm.normalize(batch["actions"].to(dev).float())
    if extra and hasattr(policy, "_cond"):
        cond = policy._cond(vis, batch["state"], batch["text"], None, extra.get("points"), extra.get("path"))
    else:
        lt, lm = policy._language_for(batch["text"], None, vis.shape[0])
        cond = policy.build_cond(vis, lt, lm, batch["state"])
    B = x.shape[0]
    z = torch.zeros_like(x).to(policy.compute_dtype)
    u = policy.head(z, torch.zeros(B, device=x.device), torch.ones(B, device=x.device), cond)
    sq = (-u.float() - x).pow(2).mean()
    return sq, {"raw_mse": float(sq.detach())}


def forward_loss(policy, batch: dict, dev: str, use_bf16: bool, done_head=None, done_weight: float = 0.1,
                 objective: str = "meanflow"):
    from dem import preprocess_images
    images = preprocess_images(batch["images"].to(dev, non_blocking=True), size=256)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        vis = policy.encode_vision(images)
    extra = {k: batch[k].to(dev) for k in ("points", "path") if k in batch}
    if objective == "mse":
        loss, aux = mse_loss(policy, vis.float(), batch, dev, extra)
    else:
        loss, aux = policy.loss(None, batch["state"], batch["actions"], texts=batch["text"], vis_tokens=vis.float(),
                                **extra)
    if done_head is not None:
        pred = done_head(vis, batch["state"].to(dev))
        dl = torch.nn.functional.mse_loss(pred, batch["progress"].to(dev).float())
        loss = loss + done_weight * dl
        aux = dict(aux, done_mse=float(dl.detach()))
    return loss, aux
