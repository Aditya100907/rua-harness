"""Skill command v0: what the LLM sends, and what the policy receives.

Two channels, kept apart on purpose:

* semantic channel -> short text -> the frozen language encoder (32-token limit in DEM)
* geometric channel -> 3D points in the robot base frame -> a small MLP / vision-token tags

Numbers never go through the language encoder. Names never go through the MLP.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional, Sequence

# skill -> (required fields, optional fields). "object" is what the gripper touches.
SKILLS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "pick_place": (("object", "target"), ("source",)),
    "open":       (("object",), ()),
    "close":      (("object",), ()),
    "slide":      (("object", "direction"), ()),
    "press":      (("object",), ("direction",)),
    "turn":       (("object", "direction"), ()),
    "insert":     (("object", "target"), ()),
    "navigate":   (("target",), ()),
}
DIRECTIONS = ("on", "off", "in", "out", "up", "down", "left", "right")


@dataclass
class Ref:
    """A thing in the scene: a name (semantic) and optionally where to act on it (geometric)."""
    name: str
    point: Optional[Sequence[float]] = None  # [x, y, z] metres, robot base frame


@dataclass
class Command:
    skill: str
    object: Optional[Ref] = None
    target: Optional[Ref] = None
    source: Optional[str] = None      # name only, e.g. "counter"
    direction: Optional[str] = None
    max_seconds: float = 30.0         # harness stops the skill after this long

    @classmethod
    def from_json(cls, s: str | dict) -> "Command":
        d = json.loads(s) if isinstance(s, str) else dict(s)
        for k in ("object", "target"):
            if isinstance(d.get(k), str):
                d[k] = Ref(d[k])
            elif isinstance(d.get(k), dict):
                d[k] = Ref(**d[k])
        cmd = cls(**d)
        validate(cmd)
        return cmd


def validate(cmd: Command) -> None:
    if cmd.skill not in SKILLS:
        raise ValueError(f"unknown skill {cmd.skill!r}; expected one of {sorted(SKILLS)}")
    required, optional = SKILLS[cmd.skill]
    for f in required:
        if getattr(cmd, f) is None:
            raise ValueError(f"skill {cmd.skill!r} needs field {f!r}")
    for f in ("object", "target", "source", "direction"):
        if getattr(cmd, f) is not None and f not in required + optional:
            raise ValueError(f"skill {cmd.skill!r} does not take field {f!r}")
    if cmd.direction is not None and cmd.direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}")
    for f in ("object", "target"):
        r = getattr(cmd, f)
        if r is not None and r.point is not None and len(r.point) != 3:
            raise ValueError(f"{f}.point must be [x, y, z]")


def to_policy_text(cmd: Command) -> str:
    """Semantic channel. Fixed field order, no punctuation, no numbers."""
    parts = ["skill", cmd.skill.replace("_", " ")]
    if cmd.object:
        parts += ["object", cmd.object.name]
    if cmd.source:
        parts += ["from", cmd.source]
    if cmd.target:
        parts += ["to", cmd.target.name]
    if cmd.direction:
        parts += ["direction", cmd.direction]
    return " ".join(parts)


def to_points(cmd: Command) -> tuple[list[list[float]], list[bool]]:
    """Geometric channel: [object point, target point] and which of them are present."""
    pts, mask = [], []
    for r in (cmd.object, cmd.target):
        has = r is not None and r.point is not None
        pts.append([float(v) for v in r.point] if has else [0.0, 0.0, 0.0])
        mask.append(has)
    return pts, mask


def to_sentence(cmd: Command) -> str:
    """English baseline, in the style of RoboCasa's own instructions."""
    o = cmd.object.name if cmd.object else ""
    t = cmd.target.name if cmd.target else ""
    return {
        "pick_place": f"Pick the {o} from the {cmd.source or 'scene'} and place it in the {t}.",
        "open": f"Open the {o}.",
        "close": f"Close the {o}.",
        "slide": f"Fully slide the {o} {cmd.direction}.",
        "press": f"Press the {o}.",
        "turn": f"Turn the {o} {cmd.direction}.",
        "insert": f"Insert the {o} into the {t}.",
        "navigate": f"Navigate to the {t}.",
    }[cmd.skill]


# What the LLM sees as a tool. It points in pixels; the harness lifts pixels to 3D.
_PIXEL_REF = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "Short noun phrase, e.g. 'red apple'."},
        "camera": {"type": "string", "enum": ["left", "right", "wrist"]},
        "uv": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2,
               "description": "Pixel [u, v] of where to act: the grasp point on an object, the handle of a door, the spot to place on."},
    },
    "required": ["name"],
}
TOOL_SCHEMA = {
    "name": "run_skill",
    "description": "Run one fast learned manipulation skill. Returns when it finishes or times out.",
    "input_schema": {
        "type": "object",
        "properties": {
            "skill": {"type": "string", "enum": sorted(SKILLS)},
            "object": _PIXEL_REF,
            "target": _PIXEL_REF,
            "source": {"type": "string"},
            "direction": {"type": "string", "enum": list(DIRECTIONS)},
            "max_seconds": {"type": "number"},
        },
        "required": ["skill"],
    },
}

# One example command per task in RoboCasa's 18-task "atomic_seen" set (the DEM paper's tasks).
ATOMIC_SEEN: dict[str, dict] = {
    "PickPlaceCounterToCabinet": {"skill": "pick_place", "object": "apple", "source": "counter", "target": "cabinet"},
    "PickPlaceCounterToStove":   {"skill": "pick_place", "object": "potato", "source": "counter", "target": "pan on stove"},
    "PickPlaceDrawerToCounter":  {"skill": "pick_place", "object": "spoon", "source": "drawer", "target": "counter"},
    "PickPlaceSinkToCounter":    {"skill": "pick_place", "object": "cup", "source": "sink", "target": "plate on counter"},
    "PickPlaceToasterToCounter": {"skill": "pick_place", "object": "toast", "source": "toaster", "target": "plate on counter"},
    "CloseFridge":               {"skill": "close", "object": "fridge door"},
    "CloseToasterOvenDoor":      {"skill": "close", "object": "toaster oven door"},
    "OpenCabinet":               {"skill": "open", "object": "cabinet door"},
    "CloseBlenderLid":           {"skill": "close", "object": "blender lid"},
    "OpenStandMixerHead":        {"skill": "open", "object": "stand mixer head"},
    "OpenDrawer":                {"skill": "open", "object": "drawer"},
    "SlideDishwasherRack":       {"skill": "slide", "object": "top dishwasher rack", "direction": "out"},
    "TurnOnMicrowave":           {"skill": "press", "object": "microwave start button"},
    "TurnOnElectricKettle":      {"skill": "press", "object": "kettle lever", "direction": "down"},
    "TurnOffStove":              {"skill": "turn", "object": "front left stove knob", "direction": "off"},
    "TurnOnSinkFaucet":          {"skill": "turn", "object": "faucet handle", "direction": "on"},
    "CoffeeSetupMug":            {"skill": "insert", "object": "mug", "target": "coffee machine"},
    "NavigateKitchen":           {"skill": "navigate", "target": "dishwasher"},
}
