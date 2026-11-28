"""Twin-distractor training data for PickPlaceCounterToCabinet.

In the stock demos the object name always picks out one object, so a point command never has
to carry the choice (docs/research_journal.md, Nov 28: the policy follows the name, not the
point). Here every demo gets a second copy of its target object (same mesh, same model file,
body "twin_main") standing still on the same counter. The instruction names a category that now
fits two objects; only the point (which stays on the true object) says which one to pick.

How one demo is built (`build_episode`):
  1. Reset to the demo scene (policy.sim.reset_to_episode) and record, from the STOCK model, the
     positions of every body at a few check frames, the target's pose at frame 0 and the
     counter height under it.
  2. Draw candidate spots on the same counter: 15 to 45 cm from the target in xy, at least 15 cm
     from the grasp point and 10 cm from every gripper position of the demo, the twin's whole
     footprint on a free surface at the target's counter height (downward rays). Orientation:
     the target's frame-0 orientation turned by a random yaw about world z.
  3. Add the twin to the XML (copy of the "obj_main" subtree, names "obj_*" -> "twin_*", a free
     joint, appended last in the worldbody) and load it (env.reset_from_xml_string).
  4. The saved states have the stock layout. Each frame's state is mapped into the new layout by
     joint name (qpos and qvel addresses); the twin is held at its spot with zero velocity.
  5. A candidate is kept when, over every frame of the demo, no twin geom touches anything but
     the counter it stands on (robot, carried object, doors), and when the twin covers at least
     MIN_LEFT_PX pixels of the left camera at frame 0 (segmentation render). Among the first few
     kept candidates the one also visible in the wrist camera at frame 0 wins, then the one with
     the most left-camera pixels.
  6. Render left, right and wrist RGB at 256 for every frame (flipped to dataset orientation) and
     JPEG them (quality 95, as policy/extract_frames.py). Check that every original body is where
     the stock replay put it (max error over the check frames, must be under 1 mm).

Outputs (base = paths.frames_base()):
  <base>/PickPlaceCounterToCabinetTwin/parts/episode_NNNNNN.npz   one per demo (jpeg bytes + meta)
  after `merge`:
  <base>/PickPlaceCounterToCabinetTwin/<cam>.bin, <cam>.idx.npy, DONE   frame store, same format
       and global frame indexing as the source task (FrameStore reads it unchanged)
  <base>/PickPlaceCounterToCabinetTwin/twin_meta.jsonl   episode -> source episode, twin pose,
       category, checks (episodes whose twin failed hold the source frames, has_twin false)
  <labels>/PickPlaceCounterToCabinetTwin.jsonl            the source labels (points on the true
       object), plus twin fields
  <runs>/twin_train.json, twin_heldout50.json             episode lists for training / evaluation

The task name PickPlaceCounterToCabinetTwin is virtual: paths.task_dir maps it to the source
task's LeRobot folder (same parquet, instructions and env), so RoboCasaDataset loads it as is.

    python -m policy.twin_data build --episodes 0 1 2 3 4 --save-png docs/img/twin
    python -m policy.twin_data build --chunk 3 --chunk-size 25
    python -m policy.twin_data merge
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from policy import paths, sim

SOURCE = "PickPlaceCounterToCabinet"
TASK = "PickPlaceCounterToCabinetTwin"
TWIN = "twin"
CAMS = ["left", "right", "wrist"]
MIN_FROM_TARGET, MAX_FROM_TARGET = 0.15, 0.45     # m, xy distance twin <-> target at frame 0
MIN_FROM_GRASP = 0.15                              # m, 3D distance twin <-> grasp point
MIN_FROM_EEF = 0.10                                # m, 3D distance twin <-> every gripper position
HEIGHT_TOL = 0.015                                 # m, support height must match the target's
MIN_LEFT_PX = 40                                   # twin pixels in the left camera at frame 0
N_CANDIDATES = 400
N_KEEP = 4                                         # kept candidates compared for wrist visibility
CHECK_EVERY = 10                                   # body-position check frames


def out_dir(base: Path | None = None) -> Path:
    return Path(base or paths.frames_base()) / TASK


def part_file(ep: int, base: Path | None = None) -> Path:
    return out_dir(base) / "parts" / f"episode_{ep:06d}.npz"


# ------------------------------------------------------------------ XML

def twin_xml(xml: str, pos, quat_wxyz, name: str = TWIN) -> str:
    """Model XML with a copy of body `obj_main` named `<name>_main`, appended last in the worldbody
    (so the stock joints keep their order). Mesh and material references are shared."""
    root = ET.fromstring(xml)
    wb = root.find("worldbody")
    src = next((b for b in wb.iter("body") if b.get("name") == "obj_main"), None)
    if src is None:
        raise ValueError("no body obj_main in the model")
    tw = copy.deepcopy(src)
    for e in tw.iter():
        n = e.get("name")
        if n is not None:
            if not n.startswith("obj_"):
                raise ValueError(f"unexpected element name {n!r} inside obj_main")
            e.set("name", name + "_" + n[4:])
    tw.set("pos", " ".join(f"{v:.6f}" for v in pos))
    tw.set("quat", " ".join(f"{v:.8f}" for v in quat_wxyz))
    wb.append(tw)
    return ET.tostring(root, encoding="unicode")


# ------------------------------------------------------------------ state mapping

def joint_layout(model) -> dict:
    """joint name -> (qpos addr, n qpos, dof addr, n dof)."""
    import mujoco
    nq = {0: 7, 1: 4, 2: 1, 3: 1}       # free, ball, slide, hinge
    nv = {0: 6, 1: 3, 2: 1, 3: 1}
    out = {}
    for j in range(model.njnt):
        t = int(model.jnt_type[j])
        out[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)] = (
            int(model.jnt_qposadr[j]), nq[t], int(model.jnt_dofadr[j]), nv[t])
    return out


class StateMap:
    """Maps a flattened stock state [time, qpos, qvel] into the twin model's layout."""

    def __init__(self, old: dict, new: dict, old_nq: int, old_nv: int, new_nq: int, new_nv: int,
                 twin_qpos: np.ndarray, twin_joint: str):
        missing = [j for j in old if j not in new]
        if missing:
            raise ValueError(f"joints missing from the twin model: {missing[:5]}")
        self.old_nq, self.old_nv, self.new_nq, self.new_nv = old_nq, old_nv, new_nq, new_nv
        qs, qd, vs, vd = [], [], [], []
        for j, (qa, nq, va, nv) in old.items():
            qa2, nq2, va2, nv2 = new[j]
            assert (nq, nv) == (nq2, nv2), j
            qs += range(qa, qa + nq); qd += range(qa2, qa2 + nq)
            vs += range(va, va + nv); vd += range(va2, va2 + nv)
        self.qs, self.qd, self.vs, self.vd = (np.asarray(x, np.int64) for x in (qs, qd, vs, vd))
        mapped = set(qd)
        tq = new[twin_joint]
        self.twin_q = np.arange(tq[0], tq[0] + tq[1])
        self.twin_v = np.arange(tq[2], tq[2] + tq[3])
        rest = set(range(new_nq)) - mapped - set(self.twin_q.tolist())
        if rest:
            raise ValueError(f"unmapped qpos slots in the twin model: {sorted(rest)[:5]}")
        self.twin_qpos = np.asarray(twin_qpos, np.float64)

    def __call__(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, np.float64)
        assert len(s) == 1 + self.old_nq + self.old_nv, (len(s), self.old_nq, self.old_nv)
        q_old, v_old = s[1:1 + self.old_nq], s[1 + self.old_nq:]
        q = np.zeros(self.new_nq); v = np.zeros(self.new_nv)
        q[self.qd] = q_old[self.qs]
        v[self.vd] = v_old[self.vs]
        q[self.twin_q] = self.twin_qpos
        v[self.twin_v] = 0.0
        return np.concatenate([[s[0]], q, v])


# ------------------------------------------------------------------ helpers on a live env

def _names(m, kind) -> list:
    import mujoco
    obj = {"body": mujoco.mjtObj.mjOBJ_BODY, "geom": mujoco.mjtObj.mjOBJ_GEOM}[kind]
    n = m.nbody if kind == "body" else m.ngeom
    return [mujoco.mj_id2name(m, obj, i) for i in range(n)]


def _subtree_bodies(m, root_name: str) -> set[int]:
    import mujoco
    r = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, root_name)
    out = {r}
    for b in range(m.nbody):
        p = b
        while p > 0:
            if p == r:
                out.add(b)
                break
            p = int(m.body_parentid[p])
    return out


def _ray_down(m, d, xy, z_from: float, exclude_body: int = -1) -> tuple[float, int]:
    """(height of the first surface hit straight down from (x, y, z_from), geom id or -1)."""
    import mujoco
    gid = np.zeros(1, np.int32)
    dist = mujoco.mj_ray(m, d, np.array([xy[0], xy[1], z_from], np.float64), np.array([0, 0, -1.0]),
                         None, 1, exclude_body, gid)
    if dist < 0:
        return -np.inf, -1
    return z_from - dist, int(gid[0])


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _seg_pixels(sim_, cam: str, geom_ids: set[int], size: int = 256) -> int:
    import mujoco
    seg = sim_.render(camera_name=sim.CAMERAS.get(cam, cam), height=size, width=size, segmentation=True)
    is_geom = seg[..., 0] == int(mujoco.mjtObj.mjOBJ_GEOM)
    return int((is_geom & np.isin(seg[..., 1], list(geom_ids))).sum())


def _bad_contacts(m, d, twin_geoms: set[int], support_bodies: set[int]) -> list:
    """Contacts of a twin geom with anything except its support (any penetration or touch)."""
    import mujoco
    bad = []
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if (g1 in twin_geoms) == (g2 in twin_geoms):
            continue
        other = g2 if g1 in twin_geoms else g1
        if int(m.geom_bodyid[other]) in support_bodies:
            continue
        if c.dist < 0.0:
            bad.append((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, other), round(float(c.dist), 4)))
    return bad


def _episode_low_dim(ds: Path, ep: int) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd
    f = next(Path(ds).glob(f"data/*/episode_{ep:06d}.parquet"))
    df = pd.read_parquet(f, columns=["observation.state", "action"])
    return np.stack(df["observation.state"].to_list()), np.stack(df["action"].to_list())


# ------------------------------------------------------------------ placement

def stock_pass(env, ds: Path, ep: int) -> dict:
    """Reset to the demo (stock model) and record what placement and the checks need."""
    states, xml, ep_meta = sim.episode_extras(ds, ep)
    sim.reset_to_episode(env, ds, ep)
    m, d = env.sim.model._model, env.sim.data._data
    T = len(states)
    check = sorted(set(range(0, T, CHECK_EVERY)) | {T - 1})
    names = _names(m, "body")
    xpos = np.zeros((len(check), m.nbody, 3))
    for k, t in enumerate(check):
        env.sim.set_state_from_flattened(states[t])
        env.sim.forward()
        xpos[k] = d.xpos
    env.sim.set_state_from_flattened(states[0])
    env.sim.forward()
    bid = env.obj_body_id["obj"]
    obj_bodies = _subtree_bodies(m, "obj_main")
    p0, q0 = d.xpos[bid].copy(), d.xquat[bid].copy()
    # counter height under the target: ray from just above the counter, target excluded (its
    # collision geom is on obj_main; the visual sub-bodies are group 1, skipped by the group mask)
    import mujoco
    gid = np.zeros(1, np.int32)
    grp = np.array([1, 0, 0, 0, 0, 0], np.uint8)          # collision geoms (group 0) only
    dist = mujoco.mj_ray(m, d, p0, np.array([0, 0, -1.0]), grp, 1, bid, gid)
    if dist < 0:
        raise RuntimeError("no support surface under the target")
    top = float(p0[2] - dist)
    support_body = int(m.geom_bodyid[gid[0]])
    gbb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "obj_reg_bbox")
    half = m.geom_size[gbb].copy() if gbb >= 0 else np.array([0.04, 0.04, 0.08])
    st16, acts = _episode_low_dim(ds, ep)
    from policy.label_commands import eef_world, gripper_events
    eef = eef_world(st16)
    closes, _ = gripper_events(acts[:, 11])
    grasp = eef[int(closes[0])] if closes else p0
    return {"states": states, "xml": env.edit_model_xml(xml), "ep_meta": ep_meta, "T": T, "check": check,
            "names": names, "xpos": xpos, "p0": p0, "q0": q0, "top": top, "support_body": support_body,
            "support_name": names[support_body], "half": half, "eef": eef, "grasp": grasp,
            "obj_bodies": obj_bodies, "layout": joint_layout(m), "nq": m.nq, "nv": m.nv,
            "cat": next((c.get("info", {}).get("cat") for c in ep_meta.get("object_cfgs", [])
                         if isinstance(c, dict) and c.get("name") == "obj"), "")}


def candidates(env, info: dict, rng: np.random.Generator, n: int = N_CANDIDATES) -> list[dict]:
    """Spots on the target's counter that pass the cheap checks (stock model, frame 0)."""
    m, d = env.sim.model._model, env.sim.data._data
    env.sim.set_state_from_flattened(info["states"][0])
    env.sim.forward()
    p0, top, hx, hy = info["p0"], info["top"], info["half"][0], info["half"][1]
    dz = p0[2] - top
    ray_from = top + 2 * info["half"][2] + 0.05
    out = []
    for _ in range(n):
        r = rng.uniform(MIN_FROM_TARGET, MAX_FROM_TARGET)
        a = rng.uniform(-np.pi, np.pi)
        xy = p0[:2] + r * np.array([np.cos(a), np.sin(a)])
        pos = np.array([xy[0], xy[1], top + dz])
        if np.linalg.norm(pos - info["grasp"]) < MIN_FROM_GRASP:
            continue
        if np.linalg.norm(info["eef"] - pos, axis=1).min() < MIN_FROM_EEF:
            continue
        yaw = rng.uniform(-np.pi, np.pi)
        quat = _qmul(_yaw_quat(yaw), info["q0"])
        # the footprint (a bit wider than the bounding box) must sit on a free surface at the target's height
        c, s = np.cos(yaw), np.sin(yaw)
        ok = True
        for fx, fy in [(0, 0), (1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)]:
            off = np.array([fx * (hx + 0.01), fy * (hy + 0.01)])
            off = np.array([c * off[0] - s * off[1], s * off[0] + c * off[1]])
            z, g = _ray_down(m, d, xy + off, ray_from)
            if g < 0 or abs(z - top) > HEIGHT_TOL:
                ok = False
                break
        if ok:
            out.append({"pos": pos, "quat": quat, "yaw": float(yaw), "r_xy": float(r)})
    return out


def load_twin(env, info: dict, cand: dict) -> tuple[StateMap, set, set]:
    """Load the model with the twin; returns (state map, twin geom ids, support body ids)."""
    xml2 = twin_xml(info["xml"], cand["pos"], cand["quat"])
    env.reset_from_xml_string(xml2)
    env.sim.reset()
    m = env.sim.model._model
    new = joint_layout(m)
    smap = StateMap(info["layout"], new, info["nq"], info["nv"], m.nq, m.nv,
                    np.concatenate([cand["pos"], cand["quat"]]), f"{TWIN}_joint0")
    tb = _subtree_bodies(m, f"{TWIN}_main")
    tg = {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in tb}
    names = _names(m, "body")
    support = {names.index(info["support_name"])}
    return smap, tg, support


def evaluate_candidate(env, info: dict, smap: StateMap, cand: dict, tg: set, support: set) -> dict:
    """Contacts over every frame and visibility at frame 0 for one twin spot."""
    m, d = env.sim.model._model, env.sim.data._data
    smap.twin_qpos = np.concatenate([cand["pos"], cand["quat"]])
    vis = {}
    for t in range(info["T"]):
        env.sim.set_state_from_flattened(smap(info["states"][t]))
        env.sim.forward()
        bad = _bad_contacts(m, d, tg, support)
        if bad:
            return {"ok": False, "why": f"contact at frame {t}: {bad[:2]}"}
        if t == 0:
            vis = {c: _seg_pixels(env.sim, c, tg) for c in CAMS}
            if vis["left"] < MIN_LEFT_PX:
                return {"ok": False, "why": f"left pixels {vis['left']}", "px0": vis}
    return {"ok": True, "px0": vis}


def place_twin(env, info: dict, rng: np.random.Generator) -> tuple[dict, StateMap, set]:
    cands = candidates(env, info, rng)
    if not cands:
        raise RuntimeError("no free counter spot passed the distance and support checks")
    smap, tg, support = load_twin(env, info, cands[0])
    kept, tried, why = [], 0, {}
    for c in cands:
        tried += 1
        r = evaluate_candidate(env, info, smap, c, tg, support)
        if r["ok"]:
            kept.append({**c, "px0": r["px0"]})
            if len(kept) >= N_KEEP:
                break
        else:
            k = r["why"].split(" ")[0]
            why[k] = why.get(k, 0) + 1
        if tried >= 60 and kept:
            break
        if tried >= 150:
            break
    if not kept:
        raise RuntimeError(f"no candidate passed ({len(cands)} spots, tried {tried}: {why})")
    best = max(kept, key=lambda c: (c["px0"]["wrist"] > 0, c["px0"]["left"]))
    best.update(n_candidates=len(cands), n_tried=tried, n_kept=len(kept), rejects=why)
    smap.twin_qpos = np.concatenate([best["pos"], best["quat"]])
    return best, smap, tg


# ------------------------------------------------------------------ one demo

def build_episode(env, ds: Path, ep: int, size: int = 256, quality: int = 95, render: bool = True) -> dict:
    """Place the twin, render every frame, run the checks. Returns arrays for the part file."""
    from policy.extract_frames import encode_jpeg
    t0 = time.time()
    info = stock_pass(env, ds, ep)
    t_stock = time.time() - t0
    rng = np.random.default_rng(10_000 + ep)
    best, smap, tg = place_twin(env, info, rng)
    t_place = time.time() - t0 - t_stock
    m, d = env.sim.model._model, env.sim.data._data
    names2 = _names(m, "body")
    idx2 = np.array([names2.index(n) for n in info["names"]])
    err = 0.0
    T = info["T"]
    blobs = {c: [] for c in CAMS}
    twin_px_wrist = []
    first_rgb = {}
    check = set(info["check"])
    ck = {t: k for k, t in enumerate(info["check"])}
    for t in range(T):
        env.sim.set_state_from_flattened(smap(info["states"][t]))
        env.sim.forward()
        if t in check:
            err = max(err, float(np.abs(d.xpos[idx2] - info["xpos"][ck[t]]).max()))
            if t % (CHECK_EVERY * 3) == 0:
                twin_px_wrist.append(_seg_pixels(env.sim, "wrist", tg))
        if render:
            for c in CAMS:
                rgb = np.ascontiguousarray(env.sim.render(camera_name=sim.CAMERAS[c], height=size, width=size)[::-1])
                if t == 0:
                    first_rgb[c] = rgb
                blobs[c].append(encode_jpeg(rgb, quality))
    twin_pos_end = d.xpos[names2.index(f"{TWIN}_main")].copy()
    meta = {"episode": ep, "source_task": SOURCE, "source_episode": ep, "has_twin": True,
            "category": info["cat"], "twin_pos": [round(float(v), 5) for v in best["pos"]],
            "twin_quat_wxyz": [round(float(v), 6) for v in best["quat"]], "twin_yaw": round(best["yaw"], 4),
            "target_pos0": [round(float(v), 5) for v in info["p0"]],
            "grasp_world": [round(float(v), 5) for v in info["grasp"]],
            "dist_target_xy_cm": round(100 * best["r_xy"], 1),
            "dist_grasp_cm": round(100 * float(np.linalg.norm(best["pos"] - info["grasp"])), 1),
            "min_dist_eef_cm": round(100 * float(np.linalg.norm(info["eef"] - best["pos"], axis=1).min()), 1),
            "support": info["support_name"], "counter_top_z": round(info["top"], 4),
            "twin_px0": best["px0"], "twin_px_wrist_samples": twin_px_wrist,
            "max_body_err_mm": round(1000 * err, 4),
            "twin_drift_mm": round(1000 * float(np.linalg.norm(twin_pos_end - best["pos"])), 4),
            "n_frames": T, "n_candidates": best["n_candidates"], "n_tried": best["n_tried"],
            "n_kept": best["n_kept"], "rejects": best["rejects"],
            "seconds_stock_place_total": [round(t_stock, 1), round(t_place, 1), round(time.time() - t0, 1)]}
    if err > 1e-3:
        raise RuntimeError(f"original bodies moved by {1000 * err:.3f} mm with the twin added")
    if max(best["px0"].values()) <= 0:
        raise RuntimeError("twin not visible in any camera at frame 0")
    arrays = {"meta": np.array(json.dumps(meta))}
    for c in CAMS:
        if render:
            off = np.zeros(T + 1, np.int64)
            off[1:] = np.cumsum([len(b) for b in blobs[c]])
            arrays[f"jpg_{c}"] = np.frombuffer(b"".join(blobs[c]), np.uint8)
            arrays[f"off_{c}"] = off
    return arrays, first_rgb, meta


def save_part(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.rename(path)


def save_png(first_rgb: dict, meta: dict, out: Path) -> Path:
    """Frame 0, left | right | wrist, with the twin and the target marked in the left view."""
    import cv2
    from policy.variants import base_to_pixel, world_to_base
    out.mkdir(parents=True, exist_ok=True)
    left = first_rgb["left"].copy()
    st16, _ = _episode_low_dim(paths.task_dir(SOURCE), meta["source_episode"])
    for p, col in ((meta["twin_pos"], (255, 0, 255)), (meta["target_pos0"], (0, 255, 0))):
        u, v, z = base_to_pixel(world_to_base(np.asarray(p), st16[0]), "left")
        if z > 0:
            cv2.circle(left, (int(round(u)), int(round(v))), 9, col, 2)
    img = np.concatenate([left, first_rgb["right"], first_rgb["wrist"]], 1)
    f = out / f"twin_ep{meta['episode']:03d}.png"
    cv2.imwrite(str(f), img[:, :, ::-1])
    return f


def build(episodes: list[int], base: Path | None = None, force: bool = False, png_dir: Path | None = None) -> list:
    ds = paths.task_dir(SOURCE)
    todo = [e for e in episodes if force or not part_file(e, base).exists()]
    print(f"{TASK}: {len(episodes) - len(todo)} done already, {len(todo)} to build", flush=True)
    if not todo:
        return []
    t0 = time.time()
    env = sim.make_env(ds, cams=CAMS, size=256)
    print(f"env built in {time.time() - t0:.1f}s", flush=True)
    log = out_dir(base) / "build_log.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for ep in todo:
        t1 = time.time()
        try:
            arrays, first, meta = build_episode(env, ds, ep)
        except Exception as e:      # one bad demo must not stop the chunk
            row = {"episode": ep, "error": f"{type(e).__name__}: {e}", "seconds": round(time.time() - t1, 1)}
            print(json.dumps(row), flush=True)
        else:
            save_part(part_file(ep, base), arrays)
            if png_dir:
                save_png(first, meta, png_dir)
            row = {k: meta[k] for k in ("episode", "category", "dist_target_xy_cm", "dist_grasp_cm", "min_dist_eef_cm",
                                        "twin_px0", "max_body_err_mm", "twin_drift_mm", "n_frames", "n_tried",
                                        "seconds_stock_place_total")}
            row["seconds"] = round(time.time() - t1, 1)
            print(json.dumps(row), flush=True)
        rows.append(row)
        with open(log, "a") as f:
            f.write(json.dumps(row) + "\n")
    env.close()
    return rows


# ------------------------------------------------------------------ merge into a frame store

def merge(base: Path | None = None, labels_dir: Path | None = None, runs_dir: Path | None = None,
          heldout_from: int = 452) -> dict:
    """Concatenate the part files in episode order into <cam>.bin / .idx.npy, same global frame
    indexing as the source task. Episodes without a part file get the source task's frames and
    has_twin false (and are left out of twin_train.json)."""
    from policy.extract_frames import FrameStore, episode_lengths
    ds = paths.task_dir(SOURCE)
    lens = episode_lengths(ds)
    od = out_dir(base)
    src = None
    metas = []
    for ep in range(len(lens)):
        p = part_file(ep, base)
        if p.exists():
            z = np.load(p)
            metas.append(json.loads(str(z["meta"])))
            z.close()
        else:
            metas.append({"episode": ep, "source_task": SOURCE, "source_episode": ep, "has_twin": False})
    for c in CAMS:
        offsets = [0]
        tmp = od / f"{c}.bin.tmp"
        with open(tmp, "wb") as f:
            for ep, n in enumerate(lens):
                if metas[ep]["has_twin"]:
                    z = np.load(part_file(ep, base))
                    jb, off = z[f"jpg_{c}"], z[f"off_{c}"]
                    z.close()
                    if len(off) - 1 != n:
                        raise ValueError(f"episode {ep}: {len(off) - 1} frames, parquet has {n}")
                    f.write(jb.tobytes())
                    offsets.extend((offsets[-1] + off[1:]).tolist())
                else:
                    if src is None:
                        src = FrameStore(SOURCE, CAMS, base)
                    g0 = int(np.sum(lens[:ep]))
                    for g in range(g0, g0 + n):
                        s, e = int(src.idx[c][g]), int(src.idx[c][g + 1])
                        fh = src._fh.get(c) or src._fh.setdefault(c, open(src.dir / f"{c}.bin", "rb"))
                        fh.seek(s)
                        b = fh.read(e - s)
                        f.write(b)
                        offsets.append(offsets[-1] + len(b))
        tmp.rename(od / f"{c}.bin")
        np.save(od / f"{c}.idx.npy", np.asarray(offsets, np.int64))
    with open(od / "twin_meta.jsonl", "w") as f:
        for mt in metas:
            f.write(json.dumps(mt) + "\n")
    # labels: the source labels, points on the true object, plus the twin's pose
    from policy.label_commands import load_labels
    ld = Path(labels_dir) if labels_dir else paths.frames_base().parent / "labels"
    with open(ld / f"{TASK}.jsonl", "w") as f:
        for lb, mt in zip(load_labels(SOURCE, ld), metas):
            assert lb["episode"] == mt["episode"]
            lb = dict(lb, task=TASK, source_task=SOURCE, has_twin=mt["has_twin"],
                      twin_world=mt.get("twin_pos"), twin_cat=mt.get("category"))
            f.write(json.dumps(lb) + "\n")
    ok = [m["episode"] for m in metas if m["has_twin"]]
    rd = Path(runs_dir) if runs_dir else paths.REPO.parent / "runs"
    (rd / "twin_train.json").write_text(json.dumps({TASK: [e for e in ok if e < heldout_from]}))
    (rd / "twin_heldout50.json").write_text(json.dumps({TASK: [e for e in ok if e >= heldout_from]}))
    info = {"task": TASK, "source": SOURCE, "episodes": len(lens), "frames": int(sum(lens)),
            "with_twin": len(ok), "without_twin": len(lens) - len(ok),
            "train_with_twin": sum(e < heldout_from for e in ok), "heldout_with_twin": sum(e >= heldout_from for e in ok),
            "cams": CAMS, "quality": 95}
    (od / "DONE").write_text(json.dumps(info, indent=1))
    return info


# ------------------------------------------------------------------ evaluation scenes

def load_meta(base: Path | None = None) -> dict:
    p = out_dir(base) / "twin_meta.jsonl"
    if not p.exists():
        return {}
    with open(p) as f:
        return {m["episode"]: m for m in map(json.loads, filter(str.strip, f))}


def reset_to_twin_episode(env, ds: Path, ep: int, meta: dict | None = None) -> dict:
    """Demo `ep`'s initial scene with its twin (placement from twin_meta.jsonl, else computed the
    same way as for training). Returns obs. The twin is a free body, so the policy can push it."""
    if meta is None or not meta.get("has_twin"):
        info = stock_pass(env, ds, ep)
        best, smap, _ = place_twin(env, info, np.random.default_rng(10_000 + ep))
        pos, quat = best["pos"], best["quat"]
    else:
        states, xml, ep_meta = sim.episode_extras(ds, ep)
        sim.reset_to_episode(env, ds, ep)
        m = env.sim.model._model
        info = {"xml": env.edit_model_xml(xml), "layout": joint_layout(m), "nq": m.nq, "nv": m.nv, "states": states}
        pos, quat = np.asarray(meta["twin_pos"]), np.asarray(meta["twin_quat_wxyz"])
    env.reset_from_xml_string(twin_xml(info["xml"], pos, quat))
    env.sim.reset()
    m = env.sim.model._model
    smap = StateMap(info["layout"], joint_layout(m), info["nq"], info["nv"], m.nq, m.nv,
                    np.concatenate([pos, quat]), f"{TWIN}_joint0")
    env.sim.set_state_from_flattened(smap(info["states"][0]))
    env.sim.forward()
    if hasattr(env, "update_state"):
        env.update_state()
    env.twin_pos0 = np.asarray(pos, np.float64)
    return env._get_observations(force_update=True)


def twin_position(env) -> np.ndarray:
    import mujoco
    m = env.sim.model._model
    return np.asarray(env.sim.data.body_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{TWIN}_main")])


# ------------------------------------------------------------------ CLI

def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--episodes", type=int, nargs="*", default=None)
    b.add_argument("--chunk", type=int, default=None)
    b.add_argument("--chunk-size", type=int, default=25)
    b.add_argument("--out", default=None, help="frames base (default paths.frames_base())")
    b.add_argument("--force", action="store_true")
    b.add_argument("--save-png", default=None, help="write frame 0 (left|right|wrist) PNGs here")
    mg = sub.add_parser("merge")
    mg.add_argument("--out", default=None)
    mg.add_argument("--labels", default=None)
    mg.add_argument("--runs", default=None)
    args = ap.parse_args()
    base = Path(args.out) if args.out else None
    if args.cmd == "build":
        from policy.extract_frames import episode_lengths
        n = len(episode_lengths(paths.task_dir(SOURCE)))
        if args.episodes:
            eps = args.episodes
        elif args.chunk is not None:
            eps = list(range(args.chunk * args.chunk_size, min(n, (args.chunk + 1) * args.chunk_size)))
        else:
            eps = list(range(n))
        if not eps:
            print("no episodes in this range", flush=True)
            return
        rows = build(eps, base, args.force, Path(args.save_png) if args.save_png else None)
        bad = [r for r in rows if "error" in r]
        print(f"done: {len(rows) - len(bad)} built, {len(bad)} failed", flush=True)
    else:
        print(json.dumps(merge(base, args.labels, args.runs), indent=1))


if __name__ == "__main__":
    main()
