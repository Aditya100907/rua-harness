"""Hindsight command labels: for each demo, the Command the agent should have sent.

From the recorded gripper command (action slot 11: -1 open, +1 closed) we find
the frame where the gripper first closes (grasp) and the next frame where it
opens (place). The gripper position at those frames, computed from the state
vector (base pose + end effector relative to base), becomes the command's
object point and target point. Names come from the task and `ep_meta.json`
(object category, fixture refs).

Tasks without a grasp (buttons, knobs, levers) use the gripper position at
the frame the demo first succeeds instead; demos end 17 frames after success
(seen in the replay test), so that frame is `len - 17`. NavigateKitchen uses
the final base position as the target. Each label records the method used.

Points are stored in the world frame and in the robot base frame at frame 0
(the frame `rua_harness.command.Ref.point` uses).

    python -m policy.label_commands --tasks PickPlaceCounterToCabinet --verify 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from policy import paths, sim
from policy.data import load_task_table
from rua_harness.command import ATOMIC_SEEN, Command, Ref, to_policy_text, validate

SUCCESS_TAIL = 17


def eef_world(state: np.ndarray) -> np.ndarray:
    """[..., 16] state -> [..., 3] end-effector position in the world frame. robosuite quats are xyzw."""
    base, quat, rel = state[..., 0:3], state[..., 3:7], state[..., 7:10]
    return base + R.from_quat(quat.reshape(-1, 4)).apply(rel.reshape(-1, 3)).reshape(rel.shape)


def world_to_base(p: np.ndarray, state0: np.ndarray) -> np.ndarray:
    return R.from_quat(state0[3:7]).inv().apply(np.asarray(p) - state0[0:3])


def gripper_events(grip: np.ndarray) -> tuple[list[int], list[int]]:
    g = grip > 0
    closes = list(np.where(~g[:-1] & g[1:])[0] + 1)
    opens = list(np.where(g[:-1] & ~g[1:])[0] + 1)
    if g[0]:
        closes = [0] + closes
    return closes, opens


def ep_names(ep_meta: dict) -> dict:
    cfgs = ep_meta.get("object_cfgs") or []
    if isinstance(cfgs, str):
        import ast
        cfgs = ast.literal_eval(cfgs)
    objs = {c.get("name"): c.get("info", {}).get("cat", "") for c in cfgs if isinstance(c, dict)}
    refs = ep_meta.get("fixture_refs") or {}
    if isinstance(refs, str):
        import ast
        refs = ast.literal_eval(refs)
    return {"objects": objs, "fixture_refs": refs, "lang": ep_meta.get("lang", "")}


def label_episode(task: str, states: np.ndarray, actions: np.ndarray, ep_meta: dict) -> dict:
    n = len(states)
    eef = eef_world(states)
    closes, opens = gripper_events(actions[:, 11])
    names = ep_names(ep_meta)
    tmpl = dict(ATOMIC_SEEN[task])
    obj_cat = names["objects"].get("obj", "").replace("_", " ")
    if obj_cat and task.startswith("PickPlace") or task == "CoffeeSetupMug":
        tmpl["object"] = obj_cat or tmpl.get("object")
    grasp_f = place_f = None
    if task == "NavigateKitchen":
        method = "final_base_position"
        grasp_pt, place_pt = None, states[-1, 0:3].copy()
    elif closes and (task.startswith("PickPlace") or task == "CoffeeSetupMug"):
        grasp_f = int(closes[0])
        after = [o for o in opens if o > grasp_f]
        place_f = int(after[0]) if after else n - 1
        method = "gripper_close_open"
        grasp_pt, place_pt = eef[grasp_f], eef[place_f]
    elif closes:
        grasp_f = int(closes[0])
        method = "gripper_close"
        grasp_pt, place_pt = eef[grasp_f], None
    else:
        grasp_f = max(0, n - SUCCESS_TAIL)
        method = "success_frame"
        grasp_pt, place_pt = eef[grasp_f], None
    s0 = states[0]
    to_b = (lambda p: None if p is None else [round(float(v), 5) for v in world_to_base(p, s0)])
    obj = tmpl.get("object")
    tgt = tmpl.get("target")
    cmd = Command(skill=tmpl["skill"],
                  object=Ref(obj, to_b(grasp_pt)) if obj else None,
                  target=Ref(tgt, to_b(place_pt)) if tgt else None,
                  source=tmpl.get("source"), direction=tmpl.get("direction"))
    validate(cmd)
    as_dict = json.loads(json.dumps(cmd, default=lambda o: o.__dict__))
    return {"task": task, "method": method, "grasp_frame": grasp_f, "place_frame": place_f,
            "n_frames": n, "n_close": len(closes), "n_open": len(opens),
            "grasp_world": None if grasp_pt is None else [round(float(v), 5) for v in grasp_pt],
            "place_world": None if place_pt is None else [round(float(v), 5) for v in place_pt],
            "command": as_dict, "kv_text": to_policy_text(cmd), "lang": names["lang"],
            "object_cat": obj_cat, "objects": names["objects"], "fixture_refs": names["fixture_refs"]}


def label_task(task: str, out_dir: Path) -> Path:
    tb = load_task_table(task)
    ds = tb["dir"]
    out = out_dir / f"{task}.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for ep, n in enumerate(tb["lens"]):
            g0 = tb["starts"][ep]
            st, ac = tb["states"][g0:g0 + n], tb["actions"][g0:g0 + n]
            with open(ds / "extras" / f"episode_{ep:06d}" / "ep_meta.json") as fm:
                meta = json.load(fm)
            f.write(json.dumps({"episode": ep, **label_episode(task, st, ac, meta)}) + "\n")
    return out


def load_labels(task: str, labels_dir: Path) -> list[dict]:
    with open(Path(labels_dir) / f"{task}.jsonl") as f:
        return [json.loads(l) for l in f if l.strip()]


def verify(task: str, labels: list[dict], n: int) -> list[dict]:
    """Distance from the hindsight grasp point to the object's simulator position at that frame."""
    ds = paths.task_dir(task)
    env = sim.make_env(ds, cams=None)
    rows = []
    for lb in labels[:n]:
        if lb["grasp_frame"] is None:
            continue
        ep, gf = lb["episode"], lb["grasp_frame"]
        states, _, _ = sim.episode_extras(ds, ep)
        sim.reset_to_episode(env, ds, ep)
        env.sim.set_state_from_flattened(states[gf])
        env.sim.forward()
        obs = env._get_observations(force_update=True)
        eef_sim = np.asarray(obs["robot0_eef_pos"])
        row = {"episode": ep, "grasp_frame": gf,
               "eef_state_vs_sim_cm": round(100 * float(np.linalg.norm(eef_sim - lb["grasp_world"])), 2)}
        if "obj" in getattr(env, "obj_body_id", {}):
            obj = np.asarray(env.sim.data.body_xpos[env.obj_body_id["obj"]])
            row["grasp_to_obj_cm"] = round(100 * float(np.linalg.norm(obj - lb["grasp_world"])), 2)
        rows.append(row)
        print(row, flush=True)
    env.close()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=paths.SMOKE_TASKS)
    ap.add_argument("--out", default=None)
    ap.add_argument("--verify", type=int, default=0, help="check the first N labels against the simulator")
    args = ap.parse_args()
    out_dir = Path(args.out) if args.out else paths.frames_base().parent / "labels"
    for t in args.tasks:
        p = label_task(t, out_dir)
        labels = load_labels(t, out_dir)
        methods = {}
        for lb in labels:
            methods[lb["method"]] = methods.get(lb["method"], 0) + 1
        print(f"LABELS {t}: {len(labels)} episodes -> {p}; methods {methods}", flush=True)
        if args.verify:
            rows = verify(t, labels, args.verify)
            d = np.array([r["grasp_to_obj_cm"] for r in rows if "grasp_to_obj_cm" in r])
            e = np.array([r["eef_state_vs_sim_cm"] for r in rows])
            summary = {"task": t, "n": len(rows), "eef_state_vs_sim_cm_max": float(e.max()) if len(e) else None}
            if len(d):
                summary.update(grasp_to_obj_cm_median=float(np.median(d)), p95=float(np.percentile(d, 95)),
                               max=float(d.max()), within_5cm=int((d <= 5).sum()))
            print("VERIFY", json.dumps(summary), flush=True)
            (out_dir / f"{t}_verify.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
