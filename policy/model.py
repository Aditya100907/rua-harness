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


def build_policy(fmt: str = "sentence", cams: int = 2, offline: bool = False, freeze_vision: bool = False):
    from dem import DEM, DEMConfig
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; choose from {FORMATS}")
    cfg = DEMConfig(action_dim=12, proprio_dim=16, chunk=16, num_cameras=cams, freeze_vision=freeze_vision,
                    vision_pretrained=not offline, language_pretrained=not offline)
    if fmt in ("sentence", "kv_text", "points_2d"):
        return DEM(cfg)
    from policy.variants import PointDEM
    return PointDEM(cfg, fmt)


def load_policy(path, offline: bool = False, map_location="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    fmt = ck.get("rua_format", "sentence")
    if fmt in ("sentence", "kv_text", "points_2d"):
        from dem import DEM
        return DEM.load(path, map_location=map_location)
    from policy.variants import PointDEM
    return PointDEM.load(path, map_location=map_location)


def wrap_dataset(ds, args):
    if args.format == "sentence":
        return ds
    from policy.variants import CommandDataset
    return CommandDataset(ds, args.format, labels_dir=args.labels_dir, point_noise_cm=args.point_noise_cm)


def forward_loss(policy, batch: dict, dev: str, use_bf16: bool):
    from dem import preprocess_images
    images = preprocess_images(batch["images"].to(dev, non_blocking=True), size=256)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        vis = policy.encode_vision(images)
    extra = {k: batch[k].to(dev) for k in ("points", "path") if k in batch}
    if extra:
        return policy.loss(None, batch["state"], batch["actions"], texts=batch["text"],
                           vis_tokens=vis.float(), **extra)
    return policy.loss(None, batch["state"], batch["actions"], texts=batch["text"], vis_tokens=vis.float())
