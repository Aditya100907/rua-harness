"""simlib: a neutral RoboCasa helper library shared by the agent side and the policy side.

Modules
  camera   images, metric depth, intrinsics, extrinsics, pixel <-> world (pure numpy + env helpers)
  tasks    the 18 atomic_seen tasks, horizons, env creation, success check, reset to a demo
  truth    ground-truth lookups (object positions, the fixture part each task is about).
           GRADING ONLY: the agent must never act on these.
  runner   parallel episode runner: one process and one log per episode, writes results CSVs

Image convention used everywhere in simlib: row 0 is the TOP of the image, the same as the
dataset videos. robosuite observations come bottom-row-first (OpenGL); simlib flips them.
See docs/simlib.md and docs/geometry_notes.md.
"""
from simlib.camera import (  # noqa: F401
    flip_rows, intrinsics, extrinsics_from_mujoco, metric_depth, pixel_to_world,
    world_to_pixel, depth_to_points,
)
from simlib.tasks import ATOMIC_SEEN, horizon  # noqa: F401
