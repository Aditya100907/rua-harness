"""Simulator conventions shared by the replay test, evaluation and run_skill.

Everything that must match between training data and the live simulator lives
here, so the replay test and the evaluation use the same code path:

* the environment is built from the dataset's own `extras/dataset_meta.json`
  env kwargs (controller config, object split, layouts), plus cameras;
* the 12-number LeRobot action is reordered to robosuite's order
  (`robocasa.utils.lerobot_utils.ACTION_KEY_ORDERING_HDF5`);
* the 16-number state is built from env observations in the order of
  `meta/modality.json`;
* camera images are flipped vertically, as RoboCasa's gym wrapper does, so they
  match the dataset videos.
"""
from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

import numpy as np

# LeRobot action slots (meta/modality.json) -> robosuite env action slots
LEROBOT_ACTION = {"base_motion": (0, 4), "control_mode": (4, 5), "end_effector_position": (5, 8),
                  "end_effector_rotation": (8, 11), "gripper_close": (11, 12)}
ENV_ACTION = {"end_effector_position": (0, 3), "end_effector_rotation": (3, 6), "gripper_close": (6, 7),
              "base_motion": (7, 11), "control_mode": (11, 12)}
# state vector (meta/modality.json order) <- robosuite observation keys
STATE_KEYS = [("robot0_base_pos", 3), ("robot0_base_quat", 4), ("robot0_base_to_eef_pos", 3),
              ("robot0_base_to_eef_quat", 4), ("robot0_gripper_qpos", 2)]
CAMERAS = {"left": "robot0_agentview_left", "right": "robot0_agentview_right", "wrist": "robot0_eye_in_hand"}


def lerobot_to_env_action(a: np.ndarray, binarize: bool = True) -> np.ndarray:
    """[..., 12] LeRobot order -> [..., 12] robosuite order.

    With `binarize`, the gripper and base-mode entries are snapped to -1 or +1
    by sign, as the dataset stores them (the policy predicts continuous values).
    """
    a = np.asarray(a, dtype=np.float64)
    out = np.zeros_like(a)
    for k, (s, e) in LEROBOT_ACTION.items():
        es, ee = ENV_ACTION[k]
        out[..., es:ee] = a[..., s:e]
    if binarize:
        for k in ("gripper_close", "control_mode"):
            es, _ = ENV_ACTION[k]
            out[..., es] = np.where(out[..., es] > 0, 1.0, -1.0)
    return out


def state16(obs: dict) -> np.ndarray:
    return np.concatenate([np.asarray(obs[k], dtype=np.float64).reshape(-1)[:n] for k, n in STATE_KEYS])


def images(obs: dict, cams: list[str]) -> np.ndarray:
    """uint8 [K, H, W, 3] in dataset orientation (vertically flipped from robosuite)."""
    return np.stack([np.ascontiguousarray(obs[f"{CAMERAS.get(c, c)}_image"][::-1]) for c in cams])


def dataset_env_meta(ds_dir: Path) -> dict:
    with open(Path(ds_dir) / "extras" / "dataset_meta.json") as f:
        return json.load(f)["env_args"]


def make_env(ds_dir: Path, cams: list[str] | None = None, size: int = 256, depth: bool = False,
             seed: int | None = None, render: bool = True):
    """robosuite env with the dataset's own kwargs. `cams` None means no rendering at all."""
    import robocasa  # noqa: F401  (registers environments)
    import robosuite
    from policy import determinism
    determinism.apply()            # HexaShell's counter-region fix (policy/determinism.py); RUA_DETERMINISM_FIX=0 to skip
    meta = dataset_env_meta(ds_dir)
    kw = copy.deepcopy(meta["env_kwargs"])
    kw["env_name"] = meta["env_name"]
    kw.update(has_renderer=False, ignore_done=True, use_object_obs=True, seed=seed)
    if cams and render:
        names = [CAMERAS.get(c, c) for c in cams]
        kw.update(has_offscreen_renderer=True, use_camera_obs=True, camera_names=names,
                  camera_widths=size, camera_heights=size, camera_depths=depth)
    else:
        kw.update(has_offscreen_renderer=False, use_camera_obs=False)
    return robosuite.make(**kw)


def episode_extras(ds_dir: Path, ep: int) -> tuple[np.ndarray, str, dict]:
    d = Path(ds_dir) / "extras" / f"episode_{ep:06d}"
    states = np.load(d / "states.npz")["states"]
    with gzip.open(d / "model.xml.gz", "rb") as f:
        xml = f.read().decode("utf-8")
    with open(d / "ep_meta.json") as f:
        ep_meta = json.load(f)
    return states, xml, ep_meta


def reset_to_episode(env, ds_dir: Path, ep: int) -> dict:
    """Load a demo's scene and initial simulator state (RoboCasa's playback_dataset.reset_to). Returns obs."""
    states, xml, ep_meta = episode_extras(ds_dir, ep)
    env.set_ep_meta(ep_meta)
    env.reset()
    env.reset_from_xml_string(env.edit_model_xml(xml))
    env.sim.reset()
    env.sim.set_state_from_flattened(states[0])
    env.sim.forward()
    if hasattr(env, "update_state"):
        env.update_state()
    return env._get_observations(force_update=True)


def episode_actions(ds_dir: Path, ep: int) -> np.ndarray:
    import pandas as pd
    f = next(Path(ds_dir).glob(f"data/*/episode_{ep:06d}.parquet"))
    return np.stack(pd.read_parquet(f, columns=["action"])["action"].to_list())
