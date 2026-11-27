"""Scripted baseline: a hand-written controller that reads the simulator's ground truth.

This is the "perfect command" oracle. It knows exactly where the object, the cabinet
interior, the drawer handle and the microwave button are, because it reads them from the
simulator (`simlib.truth` and the fixture's own joints and geoms). No learning, no camera.
It tells us how far a simple controller gets when the target is known exactly, so it is an
upper reference for "what if the command were perfect", not an agent result. The agent
itself never acts on ground truth (CLAUDE.md rule); this file is the one allowed exception,
and it is labelled `format = scripted` in every CSV row.

Control: proportional control on the end-effector position and orientation in the robot
base frame. The arm uses OSC_POSE with delta inputs in the base frame
(`input_ref_frame: base`): input [-1, 1] maps to 5 cm and 0.5 rad per step. The base and
torso never move (as in the demos). Each task is a short list of phases (move to a pose,
open or close the gripper, wait).

    PickPlaceCounterToCabinet: open, go above the object, descend, close, lift to shelf
        height, go in front of the lowest shelf, slide in, open, back out.
    OpenDrawer: turn the gripper to point at the drawer face (fingers closing vertically, as
        the demos do), go in front of the handle, close on it, pull along the slide joint.
    TurnOnMicrowave: close the gripper, point it at the start button, push 2 cm past the
        button face, back off 20 cm (success needs the gripper 15 cm away).

    python -m policy.scripted --tasks OpenDrawer --episodes 50 --seed0 1000 --note smoke
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from policy import paths, sim
from policy.eval import COLS, run_episode
from simlib import truth

POS_SCALE, ROT_SCALE = 0.05, 0.5          # metres / radians per unit of OSC input


def env_to_lerobot_action(a: np.ndarray) -> np.ndarray:
    """Inverse of sim.lerobot_to_env_action: robosuite order -> LeRobot order."""
    out = np.zeros(12)
    for k, (s, e) in sim.LEROBOT_ACTION.items():
        es, ee = sim.ENV_ACTION[k]
        out[s:e] = a[es:ee]
    return out


def base_frame(obs):
    """(R_world_base, p_world_base) from the observation (quaternions are xyzw)."""
    return Rotation.from_quat(obs["robot0_base_quat"]).as_matrix(), np.asarray(obs["robot0_base_pos"])


def to_base(obs, p_world):
    Rb, pb = base_frame(obs)
    return Rb.T @ (np.asarray(p_world) - pb)


def dir_to_base(obs, v_world):
    return base_frame(obs)[0].T @ np.asarray(v_world)


def rot_from_axes(approach, finger):
    """Gripper rotation (base frame) with eef z = approach and eef y = finger closing axis."""
    z = np.asarray(approach, float); z /= np.linalg.norm(z)
    y = np.asarray(finger, float); y = y - y.dot(z) * z; y /= np.linalg.norm(y)
    return np.stack([np.cross(y, z), y, z], axis=1)


R_DOWN = rot_from_axes([0, 0, -1], [0, -1, 0])   # pointing down, fingers close along base y


@dataclass
class Phase:
    name: str
    pos: np.ndarray | None = None      # eef target, base frame
    rot: np.ndarray | None = None      # eef rotation target, base frame (None = hold)
    grip: float = -1.0                 # -1 open, +1 closed
    tol: float = 0.01                  # metres
    rot_tol: float = 0.08              # radians
    max_steps: int = 120
    wait: int = 0                      # fixed number of steps (for gripper phases)
    speed: float = 1.0                 # max |input| on position
    done_fn: object = None             # optional extra stop condition fn(env) -> bool


@dataclass
class Script:
    phases: list[Phase]
    i: int = 0
    n: int = 0
    log: list = field(default_factory=list)
    hold_rot: np.ndarray | None = None

    def act(self, env, obs) -> np.ndarray | None:
        if self.i >= len(self.phases):
            return None
        ph = self.phases[self.i]
        cur_p = np.asarray(obs["robot0_base_to_eef_pos"])
        cur_R = Rotation.from_quat(obs["robot0_base_to_eef_quat"]).as_matrix()
        tgt_R = ph.rot if ph.rot is not None else (self.hold_rot if self.hold_rot is not None else cur_R)
        self.hold_rot = tgt_R
        a = np.zeros(12)
        a[11] = -1.0                                   # arm mode
        a[6] = ph.grip
        perr = (ph.pos - cur_p) if ph.pos is not None else np.zeros(3)
        rerr = Rotation.from_matrix(tgt_R @ cur_R.T).as_rotvec()
        a[0:3] = np.clip(perr / POS_SCALE, -ph.speed, ph.speed)
        a[3:6] = np.clip(rerr / ROT_SCALE, -1, 1)
        self.n += 1
        reached = (np.linalg.norm(perr) < ph.tol and np.linalg.norm(rerr) < ph.rot_tol) if not ph.wait else False
        timeout = self.n >= (ph.wait or ph.max_steps)
        extra = ph.done_fn is not None and ph.done_fn(env)
        if reached or timeout or extra:
            self.log.append((ph.name, self.n, "ok" if (reached or extra or ph.wait) else "timeout",
                             round(float(np.linalg.norm(perr)), 3)))
            self.i, self.n = self.i + 1, 0
        return a


# ------------------------------------------------------------------ task plans

def shelf_target(env, obs, fx, margin=0.10, depth_in=0.15):
    """A point on the lowest interior level of a cabinet, `depth_in` metres past its front
    plane, laterally as close to the gripper as the side margins allow. Base frame.
    Returns (target, inward unit direction, shelf bottom z)."""
    sites = fx.get_int_sites(all_points=False, relative=False)
    lvl = min(sites, key=lambda k: np.asarray(sites[k])[0][2])
    p0, px, py, _ = [to_base(obs, p) for p in np.asarray(sites[lvl])]
    u, v = px - p0, py - p0
    # the four vertical faces: (centre, along-face vector, inward vector)
    faces = [(p0 + 0.5 * u, u, v), (p0 + v + 0.5 * u, u, -v), (p0 + 0.5 * v, v, u), (p0 + u + 0.5 * v, v, -u)]
    fc, along, inward = min(faces, key=lambda f: np.linalg.norm(f[0][:2]))   # face nearest the robot
    inward = inward / np.linalg.norm(inward)
    al = along / np.linalg.norm(along)
    half = max(np.linalg.norm(along) / 2 - margin, 0.0)
    eef = np.asarray(obs["robot0_base_to_eef_pos"])
    s = np.clip((eef - fc).dot(al), -half, half)
    return fc + s * al + depth_in * inward, inward, p0[2]


def plan_pick_place_cabinet(env, obs, pitch=np.deg2rad(45), obj_name="obj"):
    """Grasp with the gripper pointing toward the cabinet and tilted `pitch` below horizontal,
    fingers closing sideways, then level the gripper (pointing straight into the cabinet) while
    rising to shelf height and slide in. Fully horizontal cannot reach objects low and close to
    the robot, the tilted pose cannot reach the shelf, and pointing down hits the cabinet with
    the arm. Levelling tilts the object in the fingers by `pitch`; we accept that."""
    obj = env.objects[obj_name]
    op = to_base(obs, truth.object_position(env, obj_name))
    h = float(np.asarray(obj.size)[2]) if hasattr(obj, "size") else 0.10
    tgt, inward, shelf_z = shelf_target(env, obs, env.cab)
    app = np.r_[inward[:2], 0.0]
    app /= np.linalg.norm(app)
    tilt = np.cos(pitch) * app + np.array([0, 0, -np.sin(pitch)])
    R_s = rot_from_axes(tilt, [-app[1], app[0], 0])       # fingers close horizontally
    R_l = rot_from_axes(app, [-app[1], app[0], 0])        # level, for the shelf
    grasp_z = op[2] + min(0.02, h / 4)
    bottom_below_eef = grasp_z - (op[2] - h / 2)          # object bottom sits this far below the eef
    place_z = shelf_z + bottom_below_eef + 0.03
    front = tgt - 0.25 * app                              # outside the cabinet, in line with the target
    grasp = np.r_[op[:2], grasp_z] + 0.01 * tilt
    pre = grasp + np.r_[-0.03 * app[:2], 0.10]           # above and slightly behind: descend onto it
    cur = np.asarray(obs["robot0_base_to_eef_pos"])
    out = -app
    return [
        Phase("orient", pos=np.r_[cur[:2] - 0.05 * app[:2], cur[2]], rot=R_s, tol=0.03, rot_tol=0.25, max_steps=50),
        Phase("pre_grasp", pos=pre, rot=R_s, tol=0.015),
        Phase("approach", pos=grasp, rot=R_s, tol=0.01, speed=0.4, max_steps=60),
        Phase("close", pos=grasp, grip=1, wait=20),
        Phase("lift", pos=grasp + [0, 0, 0.05], grip=1, tol=0.025, speed=0.5, max_steps=40),
        Phase("retreat", pos=np.r_[front[:2], grasp_z + 0.05], grip=1, tol=0.03, speed=0.5, max_steps=60),
        Phase("rise", pos=np.r_[front[:2], place_z], rot=R_l, grip=1, tol=0.03, rot_tol=0.15, speed=0.5,
              max_steps=80),
        Phase("slide_in", pos=np.r_[tgt[:2], place_z], rot=R_l, grip=1, tol=0.015, speed=0.4, max_steps=100),
        Phase("release", pos=np.r_[tgt[:2], place_z], grip=-1, wait=20),
        Phase("back_out", pos=np.r_[(tgt + 0.35 * out)[:2], place_z], grip=-1, tol=0.02, speed=0.5),
        Phase("hold", pos=np.r_[(tgt + 0.35 * out)[:2], place_z], grip=-1, wait=40),
    ]


def drawer_axes(env, fx):
    """World opening direction of the drawer (its slide joint, sign as get_door_state uses)."""
    m, d = env.sim.model, env.sim.data
    j = m.joint_name2id(f"{fx.name}_slidejoint")
    axis = d.xmat[m.jnt_bodyid[j]].reshape(3, 3) @ m.jnt_axis[j]
    return -axis                     # get_door_state: open fraction = -qpos / (0.55 * depth)


def plan_open_drawer(env, obs):
    fx = env.drawer
    part = truth.target_part(env, "OpenDrawer")
    hp = to_base(obs, part.pos)
    d_open = dir_to_base(obs, drawer_axes(env, fx))
    d_open[2] = 0
    d_open /= np.linalg.norm(d_open)
    app = -d_open                                    # gripper points into the drawer face
    # fingers close vertically on the bar; the demos roll the hand the other way for a drawer
    # on the robot's left (the last wrist joint runs out of range otherwise)
    R_h = rot_from_axes(app, [0, 0, -1 if hp[1] < 0 else 1])
    pull = 0.55 * float(fx.size[1]) + 0.04
    pre = hp - 0.10 * app
    # the bar sits at the front of the handle region box: aim 1.2 cm behind the box's front face
    m, d = env.sim.model, env.sim.data
    g = m.geom_name2id(part.names[0])
    Rg = base_frame(obs)[0].T @ d.geom_xmat[g].reshape(3, 3)
    half_depth = float((np.abs(Rg) @ m.geom_size[g]) @ np.abs(app))
    grasp = hp - (half_depth - 0.012) * app
    cur = np.asarray(obs["robot0_base_to_eef_pos"])
    high = np.r_[pre[:2], max(pre[2] + 0.15, cur[2] - 0.2)]
    return [
        # move sideways first with the starting orientation, then turn: turning first and
        # then swinging to a drawer on the robot's left gets the hand stuck
        Phase("over_handle", pos=np.r_[pre[:2], cur[2]], tol=0.03, max_steps=80),
        Phase("orient", pos=np.r_[pre[:2], cur[2]], rot=R_h, tol=0.03, rot_tol=0.25, max_steps=50),
        Phase("pre_grasp_high", pos=high, rot=R_h, tol=0.03),
        Phase("pre_grasp", pos=pre, rot=R_h, tol=0.01),
        Phase("approach", pos=grasp, rot=R_h, tol=0.01, speed=0.4),
        Phase("close", pos=grasp, rot=R_h, grip=1, wait=15),
        Phase("pull", pos=grasp + pull * d_open, rot=R_h, grip=1, tol=0.01, speed=0.4, max_steps=250),
        Phase("hold", pos=grasp + pull * d_open, rot=R_h, grip=1, wait=30),
    ]


def plan_microwave(env, obs):
    part = truth.target_part(env, "TurnOnMicrowave")
    m, d = env.sim.model, env.sim.data
    g = m.geom_name2id(part.names[0])
    gm = d.geom_xmat[g].reshape(3, 3)
    n = gm[:, int(np.argmin(m.geom_size[g]))]          # button face normal = its thinnest axis
    bp = to_base(obs, part.pos)
    n = dir_to_base(obs, n)
    if n.dot(bp) < 0:
        n = -n                                          # point away from the robot, into the panel
    n[2] = 0
    n /= np.linalg.norm(n)
    R_p = rot_from_axes(n, [0, 0, -1])
    mw = env.microwave
    return [
        Phase("orient", pos=np.asarray(obs["robot0_base_to_eef_pos"]) - 0.05 * n, rot=R_p, grip=1, tol=0.03, rot_tol=0.25,
              max_steps=60),
        Phase("pre_press", pos=bp - 0.08 * n, rot=R_p, grip=1, tol=0.01),
        Phase("press", pos=bp + 0.02 * n, rot=R_p, grip=1, tol=0.005, speed=0.3, max_steps=60,
              done_fn=lambda e: mw.get_state()["turned_on"]),
        Phase("back_off", pos=bp - 0.22 * n, rot=R_p, grip=1, tol=0.02, speed=0.6),
        Phase("hold", pos=bp - 0.22 * n, rot=R_p, grip=1, wait=40),
    ]


PLANS = {"PickPlaceCounterToCabinet": plan_pick_place_cabinet, "OpenDrawer": plan_open_drawer,
         "TurnOnMicrowave": plan_microwave}


def make_act_fn(env, task, obj_name="obj"):
    """act_fn(obs, t) -> [1, 12] LeRobot-order action, planned from ground truth at t = 0.
    `obj_name` picks which object a pick-and-place plan goes for (success is still about "obj")."""
    state = {}

    def act(obs, t):
        if "script" not in state:
            kw = {"obj_name": obj_name} if task.startswith("PickPlace") else {}
            state["script"] = Script(PLANS[task](env, obs, **kw))
        a = state["script"].act(env, obs)
        return None if a is None else env_to_lerobot_action(a)[None]
    return act, state


def run_one(env, task, seed, horizon, video=None, obs0=None, obj_name="obj"):
    if obs0 is None:
        env.rng = np.random.default_rng(seed)
        obs0 = env.reset()
    act, state = make_act_fn(env, task, obj_name)
    r = run_episode(env, task, seed, act, horizon, execute=1, video=video, reset_obs=obs0)
    r["phases"] = state["script"].log if "script" in state else []
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=paths.SMOKE_TASKS)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=1000, help="episode i uses seed seed0 + i (as policy/eval.py)")
    ap.add_argument("--video", type=int, default=0, help="save left-camera videos of the first N episodes")
    ap.add_argument("--note", default="smoke")
    ap.add_argument("--out", default=None)
    ap.add_argument("--testset", default=None, help="a policy/testsets.py set name (default: plain target split)")
    ap.add_argument("--guess-lookalike", action="store_true",
                    help="lookalike set: go for the target or its twin at random (what a category name alone allows)")
    args = ap.parse_args()

    cond = "scripted" + (f"_{args.testset}" if args.testset else "") + ("_guess" if args.guess_lookalike else "")
    out = Path(args.out) if args.out else paths.REPO / "results" / f"{dt.date.today()}_{cond}_{args.note}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    viddir = out.with_suffix("")
    commit = paths.code_commit()
    new = not out.exists()
    f = open(out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=COLS + ["testset", "went_for", "phases"], extrasaction="ignore")
    if new:
        w.writeheader()
    for task in args.tasks:
        ds = paths.task_dir(task)
        H = paths.horizon(task)
        cams = ["left", "right"] if args.video else None   # rendering does not change the sampled scene
        if args.testset:
            from policy import testsets
            env = testsets.make_env(args.testset, task, cams=cams)
        else:
            env = sim.make_env(ds, cams=cams, size=256)
        n_ok = 0
        for i in range(args.episodes):
            seed = args.seed0 + i
            env.rng = np.random.default_rng(seed)
            obs0 = env.reset()
            instr = env.get_ep_meta().get("lang", "")
            obj_name = "obj"
            if args.testset:
                from policy import testsets
                instr = testsets.instruction(args.testset, task, env, seed)
                if args.guess_lookalike and "lookalike" in env.objects:
                    obj_name = str(np.random.default_rng(seed + 104729).choice(["obj", "lookalike"]))
            frames = [] if i < args.video else None
            r = run_one(env, task, seed, H, video=frames, obs0=obs0, obj_name=obj_name)
            n_ok += r["success"]
            row = dict(task=task, episode=i, seed=seed, success=r["success"], seconds=r["seconds"], llm_calls=0,
                       input_tokens=0, output_tokens=0, cost_usd=0, code_commit=commit, checkpoint_step="",
                       format="scripted", run_id=cond, steps=r["steps"], instruction=instr,
                       testset=args.testset or "target", went_for=obj_name,
                       phases=";".join(f"{p[0]}:{p[1]}:{p[2]}" for p in r["phases"]))
            w.writerow(row)
            f.flush()
            print(json.dumps({k: row[k] for k in ("task", "episode", "success", "steps", "seconds", "phases")}),
                  flush=True)
            if frames:
                import imageio
                viddir.mkdir(parents=True, exist_ok=True)
                left = [fr[:, : fr.shape[1] // 2] for fr in frames]       # left camera only
                imageio.mimsave(viddir / f"{task}_ep{i:03d}_{'ok' if r['success'] else 'fail'}.mp4", left, fps=20)
        env.close()
        print(f"SCRIPTED {task}: {n_ok}/{args.episodes} succeeded", flush=True)
    f.close()
    print("wrote", out)


if __name__ == "__main__":
    main()
