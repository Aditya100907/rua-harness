"""Commands built from simulator ground truth at reset (F6 "perfect commands") and controlled errors (Part N).

Ground truth is used here only to play the role of a perfect agent; the policy
never reads the simulator directly. Text uses the same template and object
names as the training labels (policy/label_commands.py). Points:
  pick-and-place style tasks: grasp = the object's body position, place = the destination
  (cabinet interior centre, pan, plate, coffee receptacle); fixture tasks: grasp = the task's
  part (handle, knob, button, lever) nearest the gripper; NavigateKitchen: place = base goal.
Known mismatch: training labels use the gripper position at grasp and release, not body
origins (docs/labels_notes.md); median gap 2 to 6 cm by task.

Error modes (Part N):
  point_error_cm  shift each present point by this distance in a random direction (N1)
  distractor      grasp point on a distractor object instead of the target (N2, F6 check)
  drop            "point" (mask all points), "name" (object name -> "object"), "source" (N3)
  rename          replace the object name with this string (N4)
"""
from __future__ import annotations

import numpy as np

from policy.label_commands import ep_names
from rua_harness.command import ATOMIC_SEEN, Command, Ref, to_policy_text

PICK_LIKE = ("PickPlace", "CoffeeSetupMug")


def _is_pick(task: str) -> bool:
    return task.startswith(PICK_LIKE)


def perfect_command(env, task: str, rng: np.random.Generator | None = None, point_error_cm: float = 0.0,
                    distractor: bool = False, drop: str = "", rename: str = "") -> dict:
    from simlib.truth import object_names, object_position, target_part
    rng = rng or np.random.default_rng(0)
    names = ep_names(env.get_ep_meta())
    tmpl = dict(ATOMIC_SEEN[task])
    cat = names["objects"].get("obj", "").replace("_", " ")
    if cat and _is_pick(task):
        tmpl["object"] = cat
    part = target_part(env, task)
    eef = np.asarray(env._get_observations()["robot0_eef_pos"])
    grasp = place = None
    info = {"distractor_name": None}
    if task == "NavigateKitchen":
        place = part.pos
    elif _is_pick(task):
        grasp, place = part.object_pos, part.dest_pos
        if distractor:
            others = [n for n in object_names(env) if n != "obj" and not n.startswith(("container", "plate"))]
            if others:
                d = sorted(others, key=lambda n: np.linalg.norm(object_position(env, n) - grasp))[0]
                grasp = object_position(env, d)
                info["distractor_name"] = d
    else:
        pos = part.positions or [part.pos]
        grasp = pos[int(np.argmin([np.linalg.norm(np.asarray(p) - eef) for p in pos]))]
    pts = []
    for p in (grasp, place):
        if p is not None and point_error_cm > 0:
            v = rng.normal(size=3)
            p = np.asarray(p) + point_error_cm / 100.0 * v / np.linalg.norm(v)
        pts.append(None if p is None else np.asarray(p, dtype=np.float64))
    obj = tmpl.get("object")
    if rename and obj:
        obj = rename
    if drop in ("name", "all") and obj:
        obj = "object"
    src = None if drop in ("source", "all") else tmpl.get("source")
    cmd = Command(skill=tmpl["skill"], object=Ref(obj) if obj else None,
                  target=Ref(tmpl["target"]) if tmpl.get("target") else None,
                  source=src, direction=tmpl.get("direction"))
    no_pt = drop in ("point", "all")
    has = [pts[0] is not None and not no_pt and obj is not None,
           pts[1] is not None and not no_pt]
    if task == "NavigateKitchen":
        has = [False, pts[1] is not None and not no_pt]
    return {"kv_text": to_policy_text(cmd), "grasp_world": pts[0], "place_world": pts[1], "has": has,
            "true_grasp_world": None if part.object_pos is None and _is_pick(task) else
            (part.object_pos if _is_pick(task) else grasp), **info}


def points_tensor_input(cmd: dict, state: np.ndarray) -> np.ndarray:
    """[2, 4] base-frame points + presence, as CommandDataset produces in training."""
    from policy.variants import world_to_base
    out = np.zeros((2, 4), np.float32)
    for k, (key, h) in enumerate(zip(("grasp_world", "place_world"), cmd["has"])):
        if h:
            out[k, :3] = world_to_base(cmd[key], state)
            out[k, 3] = 1
    return out
