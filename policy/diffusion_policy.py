"""Diffusion Policy (Chi et al. 2023) as the end-to-end baseline, on our data and our evaluation loop.

The network is LeRobot's `DiffusionModel` (lerobot 0.4.4, `lerobot.policies.diffusion`): one ResNet-18
per camera (group norm, spatial-softmax keypoints, trained from scratch), a 1D conditional U-Net over a
16-step action chunk, epsilon prediction with a DDPM/DDIM noise schedule. We use the model class as is
and feed it our own samples (policy/data.py), so data, splits and normalization match the DEM runs.

LeRobot is not installed into envs/dem (its pins would move torch, huggingface_hub and others). It lives
in a `pip install --no-deps --target` overlay folder that is appended to sys.path, so every package of
envs/dem still wins (setup/setup_dp_overlay.sh). `lerobot/policies/__init__.py` imports every policy
(GR00T fails under transformers 5.x), so we register `lerobot.policies` as a bare namespace and import
only the diffusion modules; their code is unchanged.

Inputs, per control step (one observation frame, n_obs_steps = 1, as DEM):
  images  2 cameras (left agent view, wrist by default), uint8 256x256 -> [0, 1] floats; the model
          resizes to 128 and crops 90% (random crop in training, center crop at evaluation)
  state   the 16-number robot state, stage-1 normalized (per-task q01..q99 to [-1, 1], clipped to +-1)
          plus a one-hot task id (DP has no language input; the one-hot is our task conditioning)
Output: 16 stage-1 normalized actions; the evaluation loop executes the first 8.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn

FORMAT = "diffusion_policy"
LEROBOT_VERSION = "0.4.4"

DEFAULTS = dict(
    horizon=16, n_action_steps=8, resize=128, crop_ratio=0.9, down_dims=(512, 1024, 2048),
    scheduler="DDIM", train_timesteps=100, infer_steps=10, clip_sample_range=1.0, separate_cam_encoders=True,
    kernel_size=5, n_groups=8, diffusion_step_embed_dim=128, spatial_softmax_num_keypoints=32,
)


def overlay_dir() -> Path:
    if os.environ.get("RUA_DP_OVERLAY"):
        return Path(os.environ["RUA_DP_OVERLAY"])
    bouchet = Path("/nfs/roberts/project/pi_dr877/ak3288/envs/dp_overlay")
    return bouchet if bouchet.exists() else Path.home() / "apollo-lab" / "envs" / "dp_overlay"


def import_lerobot_dp():
    """Return (DiffusionConfig, DiffusionModel, PolicyFeature, FeatureType) from the pinned overlay."""
    try:
        import lerobot  # noqa: F401
    except ImportError:
        d = overlay_dir()
        if not d.exists():
            raise ImportError(f"LeRobot overlay not found at {d}; run setup/setup_dp_overlay.sh")
        sys.path.append(str(d))          # appended: envs/dem's own packages keep priority
        import lerobot  # noqa: F401
    import lerobot as _lr
    if "lerobot.policies" not in sys.modules:
        ns = types.ModuleType("lerobot.policies")
        ns.__path__ = [str(Path(_lr.__file__).parent / "policies")]
        sys.modules["lerobot.policies"] = ns
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel
    return DiffusionConfig, DiffusionModel, PolicyFeature, FeatureType


def lerobot_version() -> str:
    import_lerobot_dp()
    from importlib.metadata import version
    try:
        return version("lerobot")
    except Exception:
        return LEROBOT_VERSION


class DPPolicy(nn.Module):
    """Our wrapper: builds the LeRobot model from plain settings, trains on our batches, saves our files."""

    def __init__(self, tasks: list[str], cams: list[str], **kw):
        super().__init__()
        self.settings = dict(DEFAULTS, **{k: v for k, v in kw.items() if v is not None})
        self.settings["down_dims"] = list(self.settings["down_dims"])
        self.tasks, self.cams = list(tasks), list(cams)
        DiffusionConfig, DiffusionModel, PolicyFeature, FeatureType = import_lerobot_dp()
        s = self.settings
        state_dim = 16 + len(self.tasks)
        inputs = {"observation.state": PolicyFeature(FeatureType.STATE, (state_dim,))}
        for c in self.cams:
            inputs[f"observation.images.{c}"] = PolicyFeature(FeatureType.VISUAL, (3, 256, 256))
        cfg = DiffusionConfig(
            n_obs_steps=1, horizon=s["horizon"], n_action_steps=s["n_action_steps"],
            input_features=inputs, output_features={"action": PolicyFeature(FeatureType.ACTION, (12,))},
            resize_shape=(s["resize"], s["resize"]), crop_ratio=s["crop_ratio"], crop_is_random=True,
            use_separate_rgb_encoder_per_camera=s["separate_cam_encoders"], down_dims=tuple(s["down_dims"]),
            kernel_size=s["kernel_size"], n_groups=s["n_groups"],
            diffusion_step_embed_dim=s["diffusion_step_embed_dim"],
            spatial_softmax_num_keypoints=s["spatial_softmax_num_keypoints"],
            noise_scheduler_type=s["scheduler"], num_train_timesteps=s["train_timesteps"],
            num_inference_steps=s["infer_steps"], clip_sample=True, clip_sample_range=s["clip_sample_range"],
            device="cpu", push_to_hub=False)
        self.config = cfg
        self.model = DiffusionModel(cfg)
        self.task_index = 0

    # ---------------------------------------------------------------- inputs
    def _obs(self, images: torch.Tensor, state: torch.Tensor, task: torch.Tensor) -> dict:
        """images uint8 [B, K, H, W, 3]; state [B, 16] stage-1; task [B] long -> LeRobot batch (n_obs = 1)."""
        dev = next(self.parameters()).device
        img = images.to(dev, non_blocking=True).permute(0, 1, 4, 2, 3).float().div_(255.0)   # [B, K, 3, H, W]
        onehot = torch.nn.functional.one_hot(task.to(dev).long(), len(self.tasks)).float()
        st = torch.cat([state.to(dev).float(), onehot], -1)
        return {"observation.state": st[:, None], "observation.images": img[:, None]}

    def loss(self, batch: dict) -> torch.Tensor:
        b = self._obs(batch["images"], batch["state"], batch["task"])
        dev = b["observation.state"].device
        b["action"] = batch["actions"].to(dev).float()
        b["action_is_pad"] = ~batch["mask"].to(dev)
        return self.model.compute_loss(b)

    # ---------------------------------------------------------------- inference
    def set_task(self, task: str) -> None:
        if task not in self.tasks:
            raise KeyError(f"{task} was not a training task of this checkpoint ({self.tasks})")
        self.task_index = self.tasks.index(task)

    @torch.no_grad()
    def predict(self, images: np.ndarray, state: np.ndarray) -> np.ndarray:
        """images uint8 [K, 256, 256, 3], state [16] stage-1 normalized -> [16, 12] stage-1 actions."""
        b = self._obs(torch.as_tensor(np.ascontiguousarray(images))[None],
                      torch.as_tensor(state, dtype=torch.float32)[None], torch.tensor([self.task_index]))
        cond = self.model._prepare_global_conditioning(b)
        a = self.model.conditional_sample(1, global_cond=cond)       # full horizon; noise from torch's RNG
        return a[0].float().cpu().numpy()

    # ---------------------------------------------------------------- files
    def save(self, path) -> None:
        torch.save({"rua_format": FORMAT, "dp_settings": self.settings, "tasks": self.tasks, "cams": self.cams,
                    "lerobot_version": LEROBOT_VERSION, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path, map_location="cpu") -> "DPPolicy":
        ck = torch.load(path, map_location=map_location, weights_only=True)
        pol = cls(ck["tasks"], ck["cams"], **ck["dp_settings"])
        pol.load_state_dict(ck["state_dict"])
        return pol


class EMA:
    """Weight averaging as in the original Diffusion Policy (diffusion_policy/model/diffusion/ema_model.py):
    decay = 1 - (1 + step / inv_gamma) ** -power, clamped to [0, max_value]; power 0.75, max 0.9999."""

    def __init__(self, model: nn.Module, power: float = 0.75, inv_gamma: float = 1.0, max_value: float = 0.9999):
        import copy
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.power, self.inv_gamma, self.max_value = power, inv_gamma, max_value
        self.step = 0

    def decay(self) -> float:
        v = 1 - (1 + self.step / self.inv_gamma) ** -self.power
        return float(min(max(v, 0.0), self.max_value))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay()
        for e, p in zip(self.model.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(p.detach(), alpha=1 - d)
            else:
                e.copy_(p)
        self.step += 1
