"""`robot.run_skill(...)`: the trained DEM policy as a 15th HexaShell robot call.

HexaShell's model writes Python cells that call methods of a `robot` object. Each method is a
one-line proxy (`self._call(name, args, kwargs)`) in the sandboxed worker; the call travels to the
host, whose `dispatch` runs `_<name>` and returns a plain result object that prints as a line or a
few, problems first. Pictures go to the model through the host's `attach(kind, image, caption)`
and follow the cell's text. Errors the model should see are `CallFailed(kind, text)` with kind
ValueError, NotFound, NotExecuted or RobotError. This module mirrors that shape without importing
HexaShell:

* `RunSkillProxy.run_skill`  the worker-side method (mix it into HexaShell's `Robot`).
* `RunSkillHost`             the host side: `dispatch("run_skill", args, kwargs)` -> `Skill`.
* `Skill`                    the result: prints like HexaShell's Move / Grip / Waited.
* `ToolError(kind, text)`    maps 1:1 to HexaShell's `CallFailed(kind, text)`.

Seams (adapter arguments), so the same code runs in our env and in HexaShell's server:
`env` (a robosuite env, or `RenderedObsEnv` around HexaShell's `CasaSim`), `attach`, `log`,
`steps_left`, `pixel_size` (the picture size the model's pixels refer to: 512 in HexaShell),
`to_agent_frame` (world -> the frame the model's numbers are in: HexaShell's start frame), `runner`
(defaults to policy.run_skill.run_skill).

The model names pixels; the harness lifts them to world points from the depth picture. The
skill's `grading_only` output (the simulator's success flag) never reaches the model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from rua_harness.command import Command, Ref, to_policy_text, to_sentence, validate

#: camera word -> MuJoCo camera (same names as HexaShell's robocasa/protocol.py CAMERAS)
CAMERAS = {"left": "robot0_agentview_left", "right": "robot0_agentview_right", "wrist": "robot0_eye_in_hand"}
#: the most robot time one call may take (s); the env runs at 20 control steps per second
MAX_SECONDS = 30.0
CONTROL_HZ = 20
#: depth at or beyond this share of the far clip plane is background (no surface to act on)
FAR_SHARE = 0.95
#: checkpoint formats that read the two command points
POINT_FORMATS = ("points_3d", "points_3d_rgb", "points_2d")

#: the row to add to HexaShell's prompts/robocasa/03_api.md (motion table)
API_ROW = (
    "| `robot.run_skill(skill, object=, target=, source=, direction=, max_seconds=)` | runs a fast learned "
    "skill until it stops or `max_seconds` (default 30) of robot time pass. `skill`: 'open', 'close', "
    "'pick_place', 'slide', 'press', 'turn', 'insert' or 'navigate'. `object` / `target`: a short name, or "
    "`{'name': 'drawer', 'camera': 'left', 'uv': (x, y)}` with a pixel on the part to act on (the handle, the "
    "grasp point, the spot to place on); the pixel is lifted to 3-D from the depth picture. `source`: where the "
    "object is now ('counter'). `direction`: 'on', 'off', 'in', 'out', 'up', 'down', 'left' or 'right'. The "
    "pictures at the end are attached | Skill |"
)


class ToolError(Exception):
    """An error the model sees; `kind` is HexaShell's CallFailed kind."""

    def __init__(self, kind: str, text: str):
        super().__init__(text)
        self.kind, self.text = kind, text


# ---------------------------------------------------------------- the result the model sees

def _xyz(v) -> str:
    return "(" + ", ".join(f"{float(a):.3f}" for a in v) + ")"


_REASON_WORDS = {
    "max_seconds": "stopped at its time limit",
    "max_steps": "stopped: the episode's step budget ran out",
    # the skill stops early only on the simulator's check (docs/decisions_needed.md item 2); HexaShell's own
    # episode then ends with its usual "episode is over" message, which is where the model learns the outcome
    "success_check": "stopped before its time limit",
}


@dataclass
class Skill:
    """What `robot.run_skill` returns. Plain data: it crosses to the sandboxed worker."""

    skill: str
    words: str                     # the text the policy read (structured text or a sentence)
    reason: str                    # max_seconds, max_steps or success_check (shown as words, never as a verdict)
    steps: int
    seconds: float                 # robot time (steps / 20)
    points: list = field(default_factory=list)   # [(role, xyz in the agent's frame, cam, (x, y))]
    cams: list = field(default_factory=list)     # pictures attached
    notes: list = field(default_factory=list)
    steps_left: int | None = None
    wall_s: float = 0.0

    def __repr__(self) -> str:
        left = "" if self.steps_left is None else f"  [{self.steps_left} steps left]"
        lines = [f"Skill: {self.skill} ran {self.steps} steps ({self.seconds:.1f} s), "
                 f"{_REASON_WORDS.get(self.reason, 'stopped')}{left}",
                 f"  sent: {self.words!r}"]
        for role, xyz, cam, px in self.points:
            lines.append(f"  {role} point {_xyz(xyz)} from {cam} ({px[0]}, {px[1]}) on the depth picture")
        lines += [f"  note: {n}" for n in self.notes]
        if self.cams:
            lines.append(f"[picture attached: {', '.join(self.cams)} at the end]")
        return "\n".join(lines)

    __str__ = __repr__


# ---------------------------------------------------------------- worker side (HexaShell's proxy style)

class RunSkillProxy:
    """Mix into HexaShell's `Robot` (hexashell/benches/robocasa/proxy.py), which provides `_call`."""

    _call: Callable[[str, tuple, dict], Any]

    def run_skill(self, skill, *, object=None, target=None, source=None, direction=None, max_seconds=None):
        """Run one fast learned skill (the trained policy) until it stops or ``max_seconds`` of robot
        time pass (default 30, at most 30; uses steps).

        skill: 'open', 'close', 'pick_place', 'slide', 'press', 'turn', 'insert' or 'navigate'.
        object / target: what the hand acts on / where it goes: a short name ('drawer', 'red apple'), or
        a dict {'name': ..., 'camera': 'left' | 'right' | 'wrist', 'uv': (x, y)} with a pixel (column,
        row from the top-left corner) on the part to act on: the handle of a door, the grasp point on
        an object, the spot to place on.  The pixel is lifted to 3-D from the depth picture.
        source: where the object is now ('counter').  direction: 'on', 'off', 'in', 'out', 'up',
        'down', 'left' or 'right'.  Fields each skill takes: pick_place object + target (+ source);
        open, close: object; slide, turn: object + direction; press: object (+ direction); insert:
        object + target; navigate: target.
        Returns a Skill (.steps, .seconds, .reason, .words, .points); the pictures at the end are
        attached.  Bad fields raise ValueError (nothing moved); a pixel with no surface behind it
        raises NotFound.
        """
        return self._call("run_skill", (skill,), dict(object=object, target=target, source=source,
                                                      direction=direction, max_seconds=max_seconds))


# ---------------------------------------------------------------- adapters (pure helpers)

def checkpoint_format(checkpoint: str) -> str:
    """The command format a checkpoint was trained with (as PolicyRunner reads it)."""
    cfg = Path(checkpoint).parent / "config.json"
    return json.loads(cfg.read_text()).get("format", "sentence") if cfg.exists() else "sentence"


def lift_pixel(env, obs: dict, cam: str, uv, pixel_size: int | None = None) -> np.ndarray:
    """World point seen at pixel uv = (x, y) (column, row, top-left origin) of camera `cam`.

    Uses the obs depth when the obs has depth at `pixel_size`; otherwise renders the camera at
    `pixel_size` now (HexaShell's pictures are 512 while the policy's are 256).
    """
    from simlib.camera import near_far, render_view, views
    name = CAMERAS[cam]
    img = obs.get(f"{name}_image")
    size = int(pixel_size or (img.shape[0] if img is not None else 256))
    if obs.get(f"{name}_depth") is not None and img is not None and img.shape[0] == size:
        view = views(env, obs, [name])[name]
    else:
        view = render_view(env, name, size, size)
    u, v = (int(round(float(a))) for a in uv)
    if not (0 <= u < size and 0 <= v < size):
        raise ToolError("ValueError", f"run_skill: pixel ({u}, {v}) is outside the {size} x {size} {cam} picture")
    z = float(view.depth[v, u])
    _, far = near_far(env.sim)
    if not np.isfinite(z) or z >= FAR_SHARE * far:
        raise ToolError("NotFound", f"run_skill: no surface at {cam} ({u}, {v}) in the depth picture")
    return np.asarray(view.pixel_to_world((u, v)), dtype=np.float64)


def build_command(cmd: Command, grasp=None, place=None) -> dict:
    """The dict policy.run_skill.run_skill takes as `command` (same keys as policy.commands_gt)."""
    return {"kv_text": to_policy_text(cmd),
            "grasp_world": None if grasp is None else np.asarray(grasp, np.float64),
            "place_world": None if place is None else np.asarray(place, np.float64),
            "has": [grasp is not None, place is not None]}


def _ref(value, role: str):
    """(name, camera, (x, y)) from a name or a {'name', 'camera', 'uv'} dict."""
    if value is None:
        return None
    if isinstance(value, str):
        return value, None, None
    if not isinstance(value, dict) or not isinstance(value.get("name"), str) or not value["name"].strip():
        raise ToolError("ValueError", f"run_skill: {role} is a name or {{'name': ..., 'camera': ..., 'uv': (x, y)}}")
    cam = value.get("camera")
    uv = value.get("uv", value.get("xy"))
    if (cam is None) != (uv is None):
        raise ToolError("ValueError", f"run_skill: {role} needs both 'camera' and 'uv' to point, or neither")
    if cam is not None:
        if cam not in CAMERAS:
            raise ToolError("ValueError", f"run_skill: {role} camera is one of {', '.join(CAMERAS)}, not {cam!r}")
        try:
            u, v = (float(a) for a in uv)
        except (TypeError, ValueError):
            raise ToolError("ValueError", f"run_skill: {role} uv is a pixel (x, y)") from None
        uv = (int(round(u)), int(round(v)))
    return value["name"].strip(), cam, uv


class RenderedObsEnv:
    """An env whose step() returns obs with the policy's camera images, for an env built without
    camera obs (HexaShell's server builds RoboCasa with use_camera_obs=False and renders on demand).

    step_fn: what advances the episode one control step. In HexaShell it must be `CasaSim.step`, so
    that the episode's step count, horizon and success check stay HexaShell's own.
    """

    def __init__(self, env, cams=("left", "right"), size: int = 256, step_fn: Callable | None = None):
        self._env, self._cams, self._size, self._step_fn = env, tuple(cams), int(size), step_fn

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_observations(self, force_update: bool = False) -> dict:
        obs = dict(self._env._get_observations(force_update=force_update))
        for c in self._cams:
            # robosuite's own orientation (bottom row first), as obs["<cam>_image"] has it
            obs[f"{CAMERAS[c]}_image"] = self._env.sim.render(camera_name=CAMERAS[c], width=self._size,
                                                             height=self._size)
        return obs

    def step(self, action):
        if self._step_fn is not None:
            self._step_fn(action)
        else:
            self._env.step(action)
        return self._get_observations(force_update=True), 0.0, False, {}


# ---------------------------------------------------------------- host side (HexaShell's ArmHost style)

class RunSkillHost:
    """Answers `run_skill` calls. In HexaShell, CasaHost gets `_run_skill` from this class (or calls it)
    and adds "run_skill" to `public`; see interface.md."""

    public = ("run_skill",)

    def __init__(self, env, *, checkpoint: str, task: str | None = None, device: str | None = None,
                 attach: Callable[[str, np.ndarray, str], None] | None = None,
                 log: Callable[[dict], None] | None = None, steps_left: Callable[[], int | None] | None = None,
                 pixel_size: int | None = None, to_agent_frame: Callable[[np.ndarray], np.ndarray] | None = None,
                 obs_fn: Callable[[], dict] | None = None, runner: Callable | None = None,
                 max_seconds_cap: float = MAX_SECONDS):
        self.env = env
        self.checkpoint = str(checkpoint)
        self.format = checkpoint_format(self.checkpoint)
        self.task = task or type(env).__name__
        self.device = device
        self.attach = attach or (lambda kind, image, caption: None)
        self.log = log or (lambda event: None)
        self.steps_left = steps_left or (lambda: None)
        self.pixel_size = pixel_size
        self.to_agent_frame = to_agent_frame or (lambda p: p)
        self.obs_fn = obs_fn or (lambda: env._get_observations(force_update=True))
        if runner is None:
            from policy.run_skill import run_skill as runner
        self.runner = runner
        self.cap = float(max_seconds_cap)
        self.last: dict = {}              # harness-side record of the last call (never shown to the model)

    # -- dispatch, as ArmHost.dispatch --------------------------------------------------------------
    def dispatch(self, name: str, args: tuple, kwargs: dict) -> Any:
        if name not in self.public:
            raise ToolError("RobotError", f"robot has no '{name}'")
        try:
            return getattr(self, "_" + name)(*args, **kwargs)
        except TypeError as exc:
            if "argument" in str(exc):
                raise ToolError("TypeError", f"robot.{name}: {exc}") from None
            raise

    def _run_skill(self, skill, *, object=None, target=None, source=None, direction=None, max_seconds=None):
        obj = _ref(object, "object")
        tgt = _ref(target, "target")
        cmd = Command(skill=str(skill), object=Ref(obj[0]) if obj else None, target=Ref(tgt[0]) if tgt else None,
                      source=source, direction=direction)
        try:
            validate(cmd)
        except ValueError as exc:
            raise ToolError("ValueError", f"run_skill: {exc}") from None
        secs = MAX_SECONDS if max_seconds is None else max_seconds
        try:
            secs = float(secs)
        except (TypeError, ValueError):
            raise ToolError("ValueError", "run_skill: max_seconds is a number of seconds") from None
        if not secs > 0:
            raise ToolError("ValueError", "run_skill: max_seconds is more than 0")
        notes = []
        if secs > self.cap:
            notes.append(f"max_seconds is at most {self.cap:g}: it ran with {self.cap:g}")
            secs = self.cap
        left = self.steps_left()
        if left is not None:
            if left <= 0:
                raise ToolError("RobotError", "run_skill: no steps left in the episode")
            if secs * CONTROL_HZ > left:
                secs = left / CONTROL_HZ
                notes.append(f"limited to the {left} steps left in the episode")
        cmd.max_seconds = secs
        if self.format == "pose_path":
            raise ToolError("RobotError", "run_skill: this skill model needs a recorded path; it cannot run as a tool")

        obs = self.obs_fn()   # fresh: other calls may have moved the robot since the last skill
        lifted, shown = {}, []
        for role, ref in (("object", obj), ("target", tgt)):
            if ref is None or ref[1] is None:
                continue
            if self.format not in POINT_FORMATS:
                notes.append(f"this skill model reads words only: the {role} pixel was not used")
                continue
            p = lift_pixel(self.env, obs, ref[1], ref[2], self.pixel_size)
            lifted[role] = p
            shown.append((role, tuple(round(float(a), 4) for a in self.to_agent_frame(p)), ref[1], ref[2]))
        if self.format in POINT_FORMATS:
            for role, ref in (("object", obj), ("target", tgt)):
                if ref is not None and role not in lifted:
                    notes.append(f"no pixel for the {role}: the skill runs without its point")
        command = build_command(cmd, lifted.get("object"), lifted.get("target"))
        words = command["kv_text"] if self.format != "sentence" else to_sentence(cmd)

        try:
            out = self.runner(self.env, words, max_seconds=secs, checkpoint=self.checkpoint, task=self.task,
                              command=None if self.format == "sentence" else command, obs=obs,
                              device=self.device)
        except KeyError as exc:
            raise ToolError("RobotError", f"run_skill: the skill model was not trained for this task ({exc})") from None
        cams = []
        for cam in ("left", "right", "wrist"):
            img = out.get("images", {}).get(cam)
            if img is not None:
                self.attach("run_skill", img, f"run_skill: {cam} at the end")
                cams.append(cam)
        left_after = self.steps_left()
        result = Skill(skill=cmd.skill, words=words, reason=str(out["reason"]), steps=int(out["steps"]),
                       seconds=float(out["robot_seconds"]), points=shown, cams=cams, notes=notes,
                       steps_left=left_after, wall_s=float(out.get("wall_seconds", 0.0)))
        # the record keeps the grading flag for scoring; the model gets only `result`
        self.last = {"command": command, "words": words, "format": self.format, "max_seconds": secs,
                     "reason": out["reason"], "steps": out["steps"], "grading_only": out.get("grading_only")}
        self.log({"kind": "run_skill", **{k: v for k, v in self.last.items() if k != "command"},
                  "kv_text": command["kv_text"], "has": command["has"],
                  "grasp_world": None if command["grasp_world"] is None else command["grasp_world"].tolist(),
                  "place_world": None if command["place_world"] is None else command["place_world"].tolist()})
        return result


# ---------------------------------------------------------------- the JSON tool form (rua_harness.command.TOOL_SCHEMA)

def call_tool(host: RunSkillHost, arguments: dict) -> dict:
    """Run a TOOL_SCHEMA-shaped call (JSON arguments). Returns what goes back to the model:
    {"text": ..., "images": [(caption, image)], "error": None | kind}."""
    pics = []
    attach = host.attach
    host.attach = lambda kind, image, caption: (pics.append((caption, image)), attach(kind, image, caption))
    try:
        a = dict(arguments)
        skill = a.pop("skill", None)
        result = host.dispatch("run_skill", (skill,), a)
        return {"text": str(result), "images": pics, "error": None}
    except ToolError as exc:
        return {"text": f"{exc.kind}: {exc.text}", "images": [], "error": exc.kind}
    finally:
        host.attach = attach
