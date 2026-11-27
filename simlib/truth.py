"""Ground truth from the simulator. FOR GRADING AND DEBUGGING ONLY.

The agent never acts on anything in this module (CLAUDE.md rule). Use it to score an agent's
guess (e.g. "is the 3D point the agent sent within 5 cm of the drawer handle?"), to build masks
for figures, and to check geometry code.

Main entry points
  object_names(env), object_position(env, name), object_geom_ids(env, name)
  target_part(env, task)  -> Part: the fixture part the task is about (handle, knob, button...)
  target_spec(task)       -> the static description used to find it
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

# How to find the part that matters for each task.
#   fixture: env attribute holding the fixture (set in the task's _setup_kitchen_references)
#   part:    human name of the part
#   patterns: regexes tried in order against the fixture's element names with the fixture
#             prefix stripped. Geoms are searched first, then sites, then bodies. `{attr}` is
#             filled from env attributes (e.g. the stove knob chosen for this episode).
#   object / dest: for pick-and-place style tasks, the env.objects key to move and where to.
#   success: what env._check_success() tests (read from robocasa/environments/kitchen/atomic).
TASK_TARGETS: dict[str, dict] = {
    "CloseBlenderLid": dict(
        fixture="blender", part="blender lid (loose, placed beside the blender)",
        patterns=[r"^auxiliary_.*handle_main$", r"^auxiliary_.*_main$"],
        dest_patterns=[r"^pitcher_main$", r"^pitcher$"],
        success="blender.get_state()['lid_on_blender'] and gripper >0.15 m from lid"),
    "CloseFridge": dict(
        fixture="fxtr", part="fridge door handle",
        patterns=[r"^fridge_.*door_handle_main$", r"^fridge.*door.*handle", r"door_handle_main$"],
        success="fxtr.is_closed(): fridge door joints near 0"),
    "CloseToasterOvenDoor": dict(
        fixture="toaster_oven", part="toaster oven door handle",
        patterns=[r"^door_handle_main$", r"^door_main$"],
        success="toaster_oven.is_closed()"),
    "CoffeeSetupMug": dict(
        fixture="coffee_machine", part="coffee machine receptacle (mug goes here)",
        patterns=[r"^receptacle_place_site$"], object="obj",
        success="coffee_machine.check_receptacle_placement_for_pouring(obj) and gripper far"),
    "NavigateKitchen": dict(
        fixture="target_fixture", part="target fixture (goal = env.target_pos base pose)",
        patterns=[r"^main$"],
        success="robot base within 0.20 m of env.target_pos and facing env.target_ori"),
    "OpenCabinet": dict(
        fixture="fxtr", part="cabinet door handle(s)",
        patterns=[r"door_handle_reg_main$", r"door_handle_main$", r"handle"],
        success="fxtr.is_open(): every door joint >= 0.90 normalised"),
    "OpenDrawer": dict(
        fixture="drawer", part="drawer handle",
        patterns=[r"^door_handle_reg_main$", r"handle"],
        success="drawer.is_open(): slide joint open"),
    "OpenStandMixerHead": dict(
        fixture="stand_mixer", part="stand mixer head",
        patterns=[r"^head$"],
        success="stand_mixer.get_state()['head'] > 0.99"),
    "PickPlaceCounterToCabinet": dict(
        fixture="cab", part="cabinet interior (destination)", object="obj", dest="interior",
        success="obj_inside_of(obj, cab) and gripper far"),
    "PickPlaceCounterToStove": dict(
        fixture="stove", part="pan on the stove (destination)", object="obj",
        dest_object="container", success="obj in container (pan) and gripper far"),
    "PickPlaceDrawerToCounter": dict(
        fixture="drawer", part="object in the open drawer", object="obj", dest="any counter",
        success="obj touches any counter and gripper far"),
    "PickPlaceSinkToCounter": dict(
        fixture="sink", part="plate on the counter (destination)", object="obj",
        dest_object="container", success="obj in container, container on counter, gripper far"),
    "PickPlaceToasterToCounter": dict(
        fixture="toaster", part="plate (destination)", object="obj", dest_object="plate",
        success="obj in plate and gripper far"),
    "SlideDishwasherRack": dict(
        fixture="dishwasher", part="top dishwasher rack (rack1)",
        patterns=[r"^reg_rack1$", r"^rack1$"],
        success="dishwasher.get_state()['rack'] (rack1 joint) >= 0.95 if pull else <= 0.05"),
    "TurnOffStove": dict(
        fixture="stove", part="stove knob for the named burner",
        patterns=[r"^knob_{knob}_main$", r"^knob_{knob}$"],
        success="not stove.is_burner_on(knob)"),
    "TurnOnElectricKettle": dict(
        fixture="electric_kettle", part="kettle switch lever",
        patterns=[r"^switch_main$", r"^switch$"],
        success="electric_kettle.get_state()['turned_on']"),
    "TurnOnMicrowave": dict(
        fixture="microwave", part="microwave start button",
        patterns=[r"^start_button$"],
        success="microwave turned_on and gripper >0.15 m from start_button"),
    "TurnOnSinkFaucet": dict(
        fixture="sink", part="faucet handle",
        patterns=[r"^handle_main$", r"^handle$"],
        success="sink.get_handle_state()['water_on']"),
}


@dataclass
class Part:
    task: str
    part: str                     # human name
    fixture: str | None           # fixture name in the scene, e.g. "stack_3_main_group_2"
    fixture_class: str | None     # e.g. "Drawer"
    element: str                  # "geom", "site", "body", "object", "interior", "base_goal"
    names: list[str]              # matched element names
    pos: np.ndarray               # (3,) world position (mean of the matches)
    positions: list[np.ndarray] = field(default_factory=list)  # one per match
    how: str = ""                 # how it was found
    object: str | None = None     # env.objects key to move, if any
    object_pos: np.ndarray | None = None
    dest_pos: np.ndarray | None = None
    dest_how: str = ""


# ---------------------------------------------------------------- objects

def object_names(env) -> list[str]:
    return list(env.objects.keys())


def object_position(env, name: str = "obj") -> np.ndarray:
    """World position of an object's root body (its origin, roughly its centre)."""
    return np.array(env.sim.data.body_xpos[env.obj_body_id[name]])


def _descendant_bodies(model, root: int) -> set[int]:
    out = {root}
    for b in range(model.nbody):
        p = b
        while p > 0:
            if p in out:
                out.add(b)
                break
            p = model.body_parentid[p]
    return out


def object_geom_ids(env, name: str = "obj") -> list[int]:
    m = env.sim.model
    bodies = _descendant_bodies(m, env.obj_body_id[name])
    return [g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies]


def body_geom_ids(env, body_name: str) -> list[int]:
    m = env.sim.model
    bodies = _descendant_bodies(m, m.body_name2id(body_name))
    return [g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies]


# ---------------------------------------------------------------- fixture parts

def target_spec(task: str) -> dict:
    return TASK_TARGETS[task]


def _fixture(env, attr):
    f = getattr(env, attr, None)
    if f is None:
        f = env.fixture_refs.get(attr)
    if isinstance(f, tuple):  # some robocasa versions store (fixture, flag)
        f = f[0]
    return f


def find_elements(env, prefix: str, patterns, fill: dict | None = None):
    """Search geoms, then sites, then bodies named prefix + '_' + <match>. First hit wins.

    Returns (element, names, positions) or (None, [], []).
    """
    m, d = env.sim.model, env.sim.data
    kinds = [("geom", m.ngeom, m.geom_id2name, d.geom_xpos),
             ("site", m.nsite, m.site_id2name, d.site_xpos),
             ("body", m.nbody, m.body_id2name, d.body_xpos)]
    for pat in patterns:
        rx = re.compile(pat.format(**(fill or {})))
        for kind, n, id2name, xpos in kinds:
            hits = []
            for i in range(n):
                nm = id2name(i) or ""
                if nm.startswith(prefix + "_") and rx.search(nm[len(prefix) + 1:]):
                    hits.append((nm, np.array(xpos[i])))
            if hits:
                if kind == "body":  # body origins are often hinges: use the mean of its geoms
                    hits = [(nm, _body_center(env, nm, p)) for nm, p in hits]
                return kind, [h[0] for h in hits], [h[1] for h in hits]
    return None, [], []


def _body_center(env, body_name, fallback):
    g = body_geom_ids(env, body_name)
    return np.array(env.sim.data.geom_xpos[g]).mean(0) if g else np.array(fallback)


def _interior_center(fx):
    sites = fx.get_int_sites(all_points=True, relative=False)
    pts = np.concatenate([np.asarray(v) for v in sites.values()])
    return pts.mean(0), f"mean of {fx.name}.get_int_sites() corners ({', '.join(sites)})"


def target_part(env, task: str) -> Part:
    """Locate the part `task` is about in the current scene. Raises LookupError if not found."""
    spec = TASK_TARGETS[task]
    fx = _fixture(env, spec["fixture"])
    if fx is None:
        raise LookupError(f"{task}: env has no fixture attribute {spec['fixture']!r}")
    fill = {k: getattr(env, k) for k in ("knob",) if hasattr(env, k)}
    p = Part(task=task, part=spec["part"], fixture=fx.name, fixture_class=type(fx).__name__,
             element="", names=[], pos=np.full(3, np.nan))

    if "object" in spec:
        p.object = spec["object"]
        p.object_pos = object_position(env, p.object)

    if task == "NavigateKitchen":
        p.element, p.names = "base_goal", [fx.name]
        p.pos = np.array([*np.asarray(env.target_pos)[:2], 0.0])
        p.positions = [p.pos]
        p.how = "env.target_pos (robot base goal computed by the task); fixture = env.target_fixture"
        return p

    if "patterns" in spec:
        kind, names, pos = find_elements(env, fx.name, spec["patterns"], fill)
        if kind is None:
            raise LookupError(f"{task}: no element under {fx.name} matches {spec['patterns']}")
        p.element, p.names, p.positions = kind, names, pos
        p.pos = np.mean(pos, 0)
        p.how = f"{kind} name regex on {fx.name}_*"
    elif spec.get("dest") == "interior":
        p.element, p.names = "interior", [fx.name]
        p.pos, p.how = _interior_center(fx)
        p.positions = [p.pos]
    elif "dest_object" in spec:
        p.element, p.names = "object", [spec["dest_object"]]
        p.pos = object_position(env, spec["dest_object"])
        p.positions = [p.pos]
        p.how = f"env.objects[{spec['dest_object']!r}] body position"
    elif spec.get("dest") == "any counter":
        p.element, p.names = "object", [p.object]
        p.pos, p.positions = p.object_pos, [p.object_pos]
        p.how = "object inside the drawer (destination = any counter, no single point)"

    if p.object is not None and spec.get("dest") != "any counter":
        p.dest_pos, p.dest_how = p.pos, p.how   # the part IS the destination
    if "dest_patterns" in spec:
        kind, names, pos = find_elements(env, fx.name, spec["dest_patterns"], fill)
        if kind is not None:
            p.dest_pos, p.dest_how = np.mean(pos, 0), f"{kind} {names}"
    return p
