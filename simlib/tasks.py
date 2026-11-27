"""The 18 atomic tasks, their horizons, env creation, success check, and reset to a demo."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from simlib.camera import CAMERAS

# Copy of robocasa.utils.dataset_registry.TARGET_TASKS["atomic_seen"] (RoboCasa 425684d), so
# that fast code can use it without importing robocasa. tests/test_simlib_sim.py checks it.
ATOMIC_SEEN = (
    "CloseBlenderLid", "CloseFridge", "CloseToasterOvenDoor", "CoffeeSetupMug",
    "NavigateKitchen", "OpenCabinet", "OpenDrawer", "OpenStandMixerHead",
    "PickPlaceCounterToCabinet", "PickPlaceCounterToStove", "PickPlaceDrawerToCounter",
    "PickPlaceSinkToCounter", "PickPlaceToasterToCounter", "SlideDishwasherRack",
    "TurnOffStove", "TurnOnElectricKettle", "TurnOnMicrowave", "TurnOnSinkFaucet",
)

# Copy of ATOMIC_TASK_DATASETS[task]["horizon"] (control steps at 20 Hz). Checked by the tests.
HORIZONS = {
    "CloseBlenderLid": 900, "CloseFridge": 900, "CloseToasterOvenDoor": 450,
    "CoffeeSetupMug": 600, "NavigateKitchen": 450, "OpenCabinet": 1050, "OpenDrawer": 750,
    "OpenStandMixerHead": 450, "PickPlaceCounterToCabinet": 750,
    "PickPlaceCounterToStove": 600, "PickPlaceDrawerToCounter": 750,
    "PickPlaceSinkToCounter": 900, "PickPlaceToasterToCounter": 600,
    "SlideDishwasherRack": 450, "TurnOffStove": 750, "TurnOnElectricKettle": 450,
    "TurnOnMicrowave": 450, "TurnOnSinkFaucet": 600,
}


def horizon(task: str) -> int:
    """Episode length limit in control steps, from robocasa's dataset registry."""
    try:
        from robocasa.utils.dataset_registry import ATOMIC_TASK_DATASETS
        return int(ATOMIC_TASK_DATASETS[task]["horizon"])
    except ImportError:
        return HORIZONS[task]


def make_env(task: str, seed: int = 0, split: str = "target", size: int = 256,
             depth: bool = True, cameras=CAMERAS, **kwargs):
    """Build a RoboCasa env the way the datasets and evaluation use it.

    split="target" picks layouts/styles 1-10 and the target object instances (evaluation split).
    Returns the raw robosuite env: env.reset() -> obs dict, env.step(a) -> (obs, r, done, info),
    env.sim, env._check_success(). Images in obs are bottom-row-first; use simlib.camera.views().
    """
    import robocasa  # noqa: F401  (registers environments)
    from robocasa.utils.env_utils import create_env
    return create_env(env_name=task, split=split, seed=seed, camera_names=list(cameras),
                      camera_heights=size, camera_widths=size, camera_depths=depth, **kwargs)


def check_success(env) -> bool:
    """The task's own success test (env._check_success()). Grading only."""
    return bool(env._check_success())


def language(env) -> str:
    """The episode instruction, e.g. 'Open the left drawer.'"""
    return env.get_ep_meta()["lang"]


def dataset_dir(task: str, root: str | Path | None = None) -> Path | None:
    """Local LeRobot folder of the target/human demos for `task`, if downloaded."""
    import robocasa
    from robocasa.utils.dataset_registry import ATOMIC_TASK_DATASETS
    root = Path(root) if root else Path(robocasa.__file__).parent.parent / "datasets"
    d = root / ATOMIC_TASK_DATASETS[task]["target"]["human_path"] / "lerobot"
    return d if d.exists() else None


def reset_to_demo(env, demo_dir: str | Path, episode: int, step: int = 0):
    """Load a demo's scene (model xml + ep_meta) and set the simulator to its state at `step`.

    Mirrors robocasa/scripts/dataset_scripts/playback_dataset.py. Returns the recorded states
    array so a caller can replay further. Call env.reset() before this at least once.
    """
    ep = Path(demo_dir) / "extras" / f"episode_{episode:06d}"
    with gzip.open(ep / "model.xml.gz", "rt") as f:
        xml = f.read()
    ep_meta = json.loads((ep / "ep_meta.json").read_text())
    states = np.load(ep / "states.npz")
    states = states["states"]
    env.set_ep_meta(ep_meta)
    env.reset()
    env.reset_from_xml_string(env.edit_model_xml(xml))
    env.sim.reset()
    env.sim.set_state_from_flattened(states[step])
    env.sim.forward()
    if hasattr(env, "update_state"):
        env.update_state()
    return states
