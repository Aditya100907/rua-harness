"""LIBERO conventions shared by the replay test, the data loader and evaluation.

LIBERO runs in its own conda env (`envs/libero`: robosuite 1.4.1, MuJoCo 2.3.7), not in `envs/dem`.
Only `h5py` and `numpy` are needed to read demos; the simulator imports are lazy.

Demo layout (one hdf5 per task, `<suite>/<task_name>_demo.hdf5`, 50 demos):
  data/demo_<i>/actions             float64 [T, 7]   delta eef pos (3), delta eef rot axis-angle (3), gripper
  data/demo_<i>/states              float64 [T, S]   flattened MuJoCo state (S depends on the scene)
  data/demo_<i>/obs/agentview_rgb   uint8 [T, 128, 128, 3]  rendered upside down (OpenGL convention)
  data/demo_<i>/obs/eye_in_hand_rgb uint8 [T, 128, 128, 3]  same
  data/demo_<i>/obs/ee_states       [T, 6]  eef pos (3) + eef orientation as axis-angle (3)
  data/demo_<i>/obs/gripper_states  [T, 2]  the two finger joint positions
  data/demo_<i>/obs/joint_states    [T, 7]  arm joint positions (not used)

State used for DEM (proprio_dim = 8), the layout openpi and OpenVLA-OFT use for LIBERO:
  0:3 eef pos (world)  3:6 eef axis-angle  6:8 gripper finger qpos
  from env obs: robot0_eef_pos, quat2axisangle(robot0_eef_quat), robot0_gripper_qpos
  from demos:   ee_states (6), gripper_states (2)
Action (action_dim = 7), passed to the env unchanged: OSC_POSE delta in [-1, 1]
(0:3 position, 3:6 rotation, scaled by the controller), 6 gripper (-1 open, +1 close).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

STATE_DIM, ACTION_DIM = 8, 7
CAMERAS = {"agentview": ("agentview_image", "agentview_rgb"),
           "wrist": ("robot0_eye_in_hand_image", "eye_in_hand_rgb")}
DEFAULT_CAMS = ("agentview", "wrist")
RAW_SIZE = 128                       # LIBERO demos are rendered at 128; DEM gets them resized to 256
# per-suite step limits (OpenVLA / openpi LIBERO evaluation: longest training demo plus a margin)
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}
WAIT_STEPS = 10                      # objects settle after reset; the standard eval sends no-op actions first
NOOP = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float64)


def data_root() -> Path:
    return Path(os.environ.get("LIBERO_DATA", "/nfs/roberts/scratch/pi_dr877/ak3288/datasets/libero"))


def upright(img: np.ndarray, size: int | None = 256) -> np.ndarray:
    """LIBERO image (env obs or demo) -> upright uint8 [size, size, 3].

    MuJoCo renders with the OpenGL origin (bottom-left), so rows are reversed; flip them, as
    `policy.sim.images` does for RoboCasa. The same function is used for demos and the live env,
    so training and evaluation see identical orientation.
    """
    out = np.ascontiguousarray(np.asarray(img)[::-1])
    if size and out.shape[0] != size:
        import cv2
        out = cv2.resize(out, (size, size), interpolation=cv2.INTER_LINEAR)
    return out


def quat2axisangle(q: np.ndarray) -> np.ndarray:
    """robosuite's (x, y, z, w) quaternion -> axis-angle (robosuite.utils.transform_utils.quat2axisangle)."""
    q = np.asarray(q, dtype=np.float64).copy()
    q[3] = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(1.0 - q[3] * q[3])
    if np.isclose(den, 0.0):
        return np.zeros(3)
    return q[:3] * 2.0 * np.arccos(q[3]) / den


def state8(obs: dict) -> np.ndarray:
    return np.concatenate([np.asarray(obs["robot0_eef_pos"], np.float64),
                           quat2axisangle(obs["robot0_eef_quat"]),
                           np.asarray(obs["robot0_gripper_qpos"], np.float64)[:2]])


def images(obs: dict, cams=DEFAULT_CAMS, size: int | None = 256) -> np.ndarray:
    """uint8 [K, size, size, 3], upright."""
    return np.stack([upright(obs[CAMERAS[c][0]], size) for c in cams])


# ---------------------------------------------------------------- benchmark and env
def get_suite(suite: str):
    from libero.libero import benchmark
    return benchmark.get_benchmark_dict()[suite]()


def task_info(suite: str, task_id: int) -> dict:
    from libero.libero import get_libero_path
    s = get_suite(suite)
    t = s.get_task(task_id)
    return {"suite": suite, "task_id": task_id, "name": t.name, "language": t.language,
            "bddl": os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file),
            "demo": data_root() / suite / f"{t.name}_demo.hdf5",
            "init_states": init_states(t)}


def init_states(t) -> np.ndarray:
    """The task's 50 fixed benchmark init states. Same as `benchmark.get_task_init_states`, but with
    weights_only=False: torch >= 2.6 refuses these pickled numpy arrays by default. The files come from
    the LIBERO clone (`libero/libero/init_files`), not from a download."""
    import torch
    from libero.libero import get_libero_path
    p = os.path.join(get_libero_path("init_states"), t.problem_folder, t.init_states_file)
    return np.asarray(torch.load(p, weights_only=False))


def n_tasks(suite: str) -> int:
    return get_suite(suite).n_tasks


def make_env(bddl: str, size: int = RAW_SIZE, render: bool = True, seed: int = 0):
    """LIBERO env. `render=False` builds it without cameras (replay test)."""
    from libero.libero.envs.env_wrapper import ControlEnv, OffScreenRenderEnv
    if render:
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=size, camera_widths=size)
    else:   # OffScreenRenderEnv forces the offscreen renderer on, so use its base class
        env = ControlEnv(bddl_file_name=bddl, has_renderer=False, has_offscreen_renderer=False,
                         use_camera_obs=False)
    env.seed(seed)
    return env


def reset_to_state(env, state: np.ndarray) -> dict:
    """Reset, then set the flattened MuJoCo state (a demo's first state or a benchmark init state)."""
    env.reset()
    return env.set_init_state(np.asarray(state))


def reset_to_demo(env, demo) -> dict:
    """Load a demo's own scene XML and first state, as LIBERO's scripts/create_dataset.py does
    (reset, reset_from_xml_string(postprocessed model_file), sim.reset, set state). Returns obs."""
    import re
    from libero.libero import get_libero_path
    from libero.libero.utils.utils import postprocess_model_xml
    xml = postprocess_model_xml(demo.attrs["model_file"], {})       # fixes robosuite's own mesh paths
    # LIBERO's object assets still point at the authors' machine (".../chiliocosm/assets/...")
    xml = re.sub(r'file="[^"]*?/(?:chiliocosm|libero/libero)/assets/', f'file="{get_libero_path("assets")}/', xml)
    env.reset()
    env.reset_from_xml_string(xml)
    env.sim.reset()
    return env.set_init_state(demo["states"][0])


# ---------------------------------------------------------------- demos
def demo_keys(f) -> list[str]:
    return sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))


def demo_lowdim(path: Path, sim_states: bool = False) -> dict:
    """Per-demo states [T, 8], actions [T, 7] (and MuJoCo states if asked), from one task's hdf5."""
    import h5py
    out = {"keys": [], "state": [], "action": [], "sim_states": []}
    with h5py.File(path, "r") as f:
        for k in demo_keys(f):
            d = f["data"][k]
            out["keys"].append(k)
            out["state"].append(np.concatenate([d["obs/ee_states"][()], d["obs/gripper_states"][()]], 1))
            out["action"].append(d["actions"][()])
            if sim_states:
                out["sim_states"].append(d["states"][()])
    return out
