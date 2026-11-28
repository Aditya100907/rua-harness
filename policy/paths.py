"""Where datasets, frames and runs live on the Mac and on Bouchet.

Override any of them with environment variables:
  RUA_DATA    dataset base (the folder that holds v1.0/), default: RoboCasa's
              DATASET_BASE_PATH, else ~/apollo-lab/robocasa/datasets
  RUA_FRAMES  pre-extracted frames (policy/extract_frames.py)
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

SMOKE_TASKS = ["PickPlaceCounterToCabinet", "OpenDrawer", "TurnOnMicrowave"]
ATOMIC_SEEN = [
    "CloseBlenderLid", "CloseFridge", "CloseToasterOvenDoor", "CoffeeSetupMug", "NavigateKitchen",
    "OpenCabinet", "OpenDrawer", "OpenStandMixerHead", "PickPlaceCounterToCabinet",
    "PickPlaceCounterToStove", "PickPlaceDrawerToCounter", "PickPlaceSinkToCounter",
    "PickPlaceToasterToCounter", "SlideDishwasherRack", "TurnOffStove", "TurnOnElectricKettle",
    "TurnOnMicrowave", "TurnOnSinkFaucet",
]
# Virtual tasks: same demos (parquet, instructions, env) as the source task, their own frame
# store and labels. PickPlaceCounterToCabinetTwin: every demo re-rendered with a twin of the
# target object on the counter (policy/twin_data.py).
VIRTUAL_TASKS = {"PickPlaceCounterToCabinetTwin": "PickPlaceCounterToCabinet"}
REPO = Path(__file__).resolve().parents[1]
BOUCHET_SCRATCH = Path("/nfs/roberts/scratch/pi_dr877/ak3288")


def data_base() -> Path:
    if os.environ.get("RUA_DATA"):
        return Path(os.environ["RUA_DATA"])
    if BOUCHET_SCRATCH.exists():
        return BOUCHET_SCRATCH / "datasets"
    return Path.home() / "apollo-lab" / "robocasa" / "datasets"


def frames_base() -> Path:
    if os.environ.get("RUA_FRAMES"):
        return Path(os.environ["RUA_FRAMES"])
    if BOUCHET_SCRATCH.exists():
        return BOUCHET_SCRATCH / "frames"
    return Path.home() / "apollo-lab" / "runs" / "frames"


def source_task(task: str) -> str:
    """The RoboCasa task behind `task` (itself unless it is a virtual task)."""
    return VIRTUAL_TASKS.get(task, task)


def task_dir(task: str, base: Path | None = None) -> Path:
    """The `lerobot/` folder of a task's target human demos (the source task's, for a virtual task)."""
    task = source_task(task)
    base = Path(base) if base else data_base()
    hits = sorted((base / "v1.0" / "target" / "atomic" / task).glob("*/lerobot"))
    if not hits:
        raise FileNotFoundError(f"no target demos for {task} under {base}")
    return hits[-1]


def horizon(task: str) -> int:
    """Episode step limit RoboCasa uses for a task."""
    from robocasa.utils.dataset_registry import ATOMIC_TASK_DATASETS
    return int(ATOMIC_TASK_DATASETS[source_task(task)]["horizon"])


def code_commit() -> str:
    # The Bouchet copy is synced by rsync without .git (scripts/sync_bouchet.sh writes COMMIT), but it still
    # holds a stale .git from the first setup (b180b55), so COMMIT must win over git when it exists.
    # Before Nov 28 14:15, cluster results recorded "b180b55-dirty" because of this.
    p = REPO / "COMMIT"
    if p.exists():
        return p.read_text().strip()
    try:
        c = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        if c:
            return c + ("-dirty" if dirty else "")
    except Exception:
        pass
    p = REPO / "COMMIT"            # rsync excludes .git; the sync script writes this file
    return p.read_text().strip() if p.exists() else "unknown"
