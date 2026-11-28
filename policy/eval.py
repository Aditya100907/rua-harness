"""Evaluate a DEM checkpoint on RoboCasa target-split episodes.

The environment comes from `policy.sim.make_env` (the dataset's own env kwargs,
as in the replay test), reset with a per-episode seed, so kitchens come from the
target layouts 1..10 and the target object split. Each control cycle feeds the
left and right agent views (flipped to dataset orientation) and the 16-number
state, predicts 16 actions and executes the first 8, until success or the task's
horizon from `dataset_registry.py`.

One CSV row per episode in `results/` (columns of results/README.md, plus
checkpoint_step, format, run_id, steps, instruction).

    python -m policy.eval --ckpt runs/x/ckpt_0020000.pt --tasks OpenDrawer --episodes 50
    python -m policy.eval --wiring --tasks OpenDrawer --episodes 10      # D1 test 1: demo actions, no policy
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np

from policy import paths, sim

COLS = ["task", "episode", "seed", "success", "seconds", "llm_calls", "input_tokens", "output_tokens",
        "cost_usd", "code_commit", "checkpoint_step", "format", "run_id", "steps", "instruction"]


class PolicyRunner:
    """Wraps a checkpoint + norm_stats: raw obs in, raw LeRobot-order actions out."""

    def __init__(self, ckpt: str, device: str = "cuda", cams=("left", "right"), nfe: int = 1, z0: str = "random"):
        import torch
        from policy.data import load_norm_stats
        from policy.model import load_policy
        self.torch, self.device, self.cams, self.nfe = torch, device, list(cams), nfe
        self.z0 = z0          # "random" (fresh noise per call) or "zero" (deterministic, the noise mean)
        self.policy = load_policy(ckpt).to(device).eval()
        ck_dir = Path(ckpt).parent
        self.stats = load_norm_stats(ck_dir / "norm_stats.json")
        cfg_p = ck_dir / "config.json"
        self.cfg = json.loads(cfg_p.read_text()) if cfg_p.exists() else {}
        if self.cfg.get("cams"):
            self.cams = list(self.cfg["cams"])
        import torch as _t
        ck_fmt = _t.load(ckpt, map_location="cpu", weights_only=True).get("rua_format")
        self.format = ck_fmt or self.cfg.get("format", "sentence")
        # sampling follows the training objective unless the caller asked for something else
        obj = self.cfg.get("objective", "meanflow")
        self.objective = obj
        if obj == "mse" and z0 == "random":
            self.z0 = "zero"                  # regression: the action is -u(0); noise would be wrong
        if obj == "fm" and nfe == 1:
            self.nfe = 10                     # plain flow matching needs several integration steps
        if ck_fmt and self.cfg.get("format") and ck_fmt != self.cfg["format"]:
            raise ValueError(f"checkpoint format {ck_fmt} != config.json format {self.cfg['format']}")
        self.quant = None
        self.env = None

    def set_task(self, task: str, instruction: str, env=None, cmd_opts: dict | None = None,
                 rng: np.random.Generator | None = None) -> dict:
        """Returns the command used (for formats other than `sentence`, built from ground truth at reset)."""
        from policy.data import task_quantiles
        if task not in self.stats["per_task"]:
            raise KeyError(f"{task} was not in this checkpoint's training tasks")
        self.quant = task_quantiles(self.stats, task)
        self.cmd = None
        self.env = env
        if self.format == "sentence":
            self.policy.set_instruction(instruction)
            return {"text": instruction}
        if self.format == "pose_path":
            # demo-scene evaluation: the "perfect" path is the held-out demo's own future (time-indexed)
            self.cmd = {"kv_text": (cmd_opts or {}).get("kv_text", instruction)}
            self.policy.set_instruction(self.cmd["kv_text"])
            return {"text": self.cmd["kv_text"]}
        from policy.commands_gt import perfect_command
        self.cmd = perfect_command(env, task, rng=rng, **(cmd_opts or {}))
        self.policy.set_instruction(self.cmd["kv_text"])
        return {"text": self.cmd["kv_text"], **self.cmd}

    def set_demo(self, states: np.ndarray, t_ref: list):
        """pose_path: recorded 16-number states of the demo whose scene is being replayed."""
        self.demo_states, self.t_ref = states, t_ref

    def demo_path(self, obs: dict) -> np.ndarray:
        from scipy.spatial.transform import Rotation as R
        from policy.variants import PATH_LEN, PATH_STRIDE
        t = self.t_ref[0]
        st = sim.state16(obs)
        n = len(self.demo_states)
        fut = self.demo_states[np.minimum(t + PATH_STRIDE * np.arange(1, PATH_LEN + 1), n - 1)]
        p_w = fut[:, 0:3] + R.from_quat(fut[:, 3:7]).apply(fut[:, 7:10])
        p_b = R.from_quat(st[3:7]).inv().apply(p_w - st[0:3])
        q_b = R.from_quat(st[3:7]).inv() * (R.from_quat(fut[:, 3:7]) * R.from_quat(fut[:, 10:14]))
        return np.concatenate([p_b, q_b.as_quat()], 1).astype(np.float32)

    def _images(self, obs: dict) -> np.ndarray:
        if self.format == "points_3d":      # point map from live depth, in the current base frame
            from policy.variants import pointmap_to_image
            from simlib.camera import views
            if self.env is None:
                raise RuntimeError("points_3d needs the env (build it with depth=True)")
            v = views(self.env, obs, [sim.CAMERAS[c] for c in self.cams])
            st = sim.state16(obs)
            from scipy.spatial.transform import Rotation as R
            out = []
            for c in self.cams:
                pm = v[sim.CAMERAS[c]].point_map()                      # [H, W, 3] world
                pb = R.from_quat(st[3:7]).inv().apply(pm.reshape(-1, 3) - st[0:3]).reshape(pm.shape)
                out.append(pointmap_to_image(pb.transpose(2, 0, 1)))
            return np.stack(out)
        imgs = sim.images(obs, self.cams)
        if self.format == "points_2d" and self.cmd is not None:
            from policy.variants import CAM_IN_BASE, base_to_pixel, draw_points, world_to_base
            st = sim.state16(obs)
            out = []
            for k, cam in enumerate(self.cams):
                if cam in CAM_IN_BASE:
                    uv = [base_to_pixel(world_to_base(self.cmd[key], st), cam) if h else None
                          for key, h in zip(("grasp_world", "place_world"), self.cmd["has"])]
                    out.append(draw_points(imgs[k], uv))
                else:
                    out.append(imgs[k])
            imgs = np.stack(out)
        return imgs

    def act(self, obs: dict, extra: dict | None = None) -> np.ndarray:
        torch = self.torch
        from dem import preprocess_images
        qs, qa = self.quant
        imgs = preprocess_images(self._images(obs), size=256, device=self.device)[None]
        sn = qs.normalize(sim.state16(obs))
        if self.stats.get("clip"):
            sn = np.clip(sn, -self.stats["clip"], self.stats["clip"])
        st = torch.as_tensor(sn, dtype=torch.float32, device=self.device)[None]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
            vis = self.policy.encode_vision(imgs)
        if extra is None and self.format == "pose_path":
            extra = {"path": torch.as_tensor(self.demo_path(obs), device=self.device)[None]}
        if extra is None and self.format in ("points_3d", "points_3d_rgb"):
            from policy.commands_gt import points_tensor_input
            extra = {"points": torch.as_tensor(points_tensor_input(self.cmd, sim.state16(obs)), device=self.device)[None]}
        extra = dict(extra or {})
        if self.z0 == "zero":
            extra["z0"] = torch.zeros(1, 16, 12, device=self.device)
        a = self.policy.act(None, st, vis_tokens=vis.float(), nfe=self.nfe, **extra)
        return qa.unnormalize(a[0].float().cpu().numpy())          # [16, 12] raw LeRobot order


def run_episode(env, task: str, seed: int, act_fn, horizon: int, execute: int = 8, video=None,
                reset_obs: dict | None = None, record: bool = False, on_step=None) -> dict:
    """act_fn(obs, t) -> [H, 12] raw LeRobot actions. Returns a result dict."""
    t0 = time.time()
    if reset_obs is None:
        env.rng = np.random.default_rng(seed)
        obs = env.reset()
    else:
        obs = reset_obs
    traj = {"eef": [], "grip_cmd": [], "obj": []} if record else None
    success, t, model_s = False, 0, 0.0
    while t < horizon and not success:
        tm = time.time()
        chunk = act_fn(obs, t)
        model_s += time.time() - tm
        if chunk is None or len(chunk) == 0:
            break
        for a in sim.lerobot_to_env_action(np.asarray(chunk)[:execute]):
            obs, _, _, _ = env.step(a)
            t += 1
            if on_step is not None:
                on_step(obs)
            if video is not None:
                video.append(np.concatenate(list(sim.images(obs, ["left", "right"])), axis=1))
            if record:
                traj["eef"].append(np.asarray(obs.get("robot0_eef_pos", np.zeros(3))))
                traj["grip_cmd"].append(a[6])
                o = env.objects.get("obj") if hasattr(env, "objects") else None
                traj["obj"].append(np.asarray(env.sim.data.body_xpos[env.obj_body_id["obj"]])
                                   if o is not None and "obj" in getattr(env, "obj_body_id", {}) else np.zeros(3))
            if env._check_success():
                success = True
                break
            if t >= horizon:
                break
    out = {"success": int(success), "steps": t, "seconds": round(time.time() - t0, 2),
           "model_seconds": round(model_s, 2)}
    if record:
        out["traj"] = {k: np.asarray(v) for k, v in traj.items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tasks", nargs="+", default=paths.SMOKE_TASKS)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=1000, help="episode i uses seed seed0 + i")
    ap.add_argument("--execute", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--wiring", action="store_true", help="D1 test 1: play demo actions through this loop")
    ap.add_argument("--video", type=int, default=2, help="save videos of the first N episodes per task")
    ap.add_argument("--save-traj", action="store_true", help="save per-episode trajectories (failure sorting)")
    ap.add_argument("--note", default="")
    ap.add_argument("--out", default=None, help="results CSV (default results/<date>_policy_only_<note>.csv)")
    ap.add_argument("--outdir", default=None, help="videos and trajectories")
    ap.add_argument("--horizon", type=int, default=0, help="override the task horizon")
    ap.add_argument("--shard", type=int, default=0, help="run episodes shard, shard + nshards, ...")
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--point-error-cm", type=float, default=0.0, help="N1: shift command points by this much")
    ap.add_argument("--distractor", action="store_true", help="N2/F6: grasp point on a distractor object")
    ap.add_argument("--distractor-rate", type=float, default=0.0,
                    help="referent axis: per episode (by seed), put the grasp point on a distractor with this probability")
    ap.add_argument("--drop", default="", choices=["", "point", "name", "source"], help="N3: drop a field")
    ap.add_argument("--rename", default="", help="N4: call the object by this name")
    ap.add_argument("--no-determinism-fix", action="store_true",
                    help="run without HexaShell's counter-region determinism fix (policy/determinism.py)")
    ap.add_argument("--z0", default="random", choices=["random", "zero"], help="sampling noise")
    ap.add_argument("--nfe", type=int, default=1, help="MeanFlow steps per action chunk")
    ap.add_argument("--demo-scenes", default="", help="JSON task -> demo episode ids: replay those initial scenes")
    args = ap.parse_args()
    if args.no_determinism_fix:
        import os as _os
        _os.environ["RUA_DETERMINISM_FIX"] = "0"
    import torch
    dev = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    runner, step, run_id, fmt = None, "", "", "sentence"
    if not args.wiring:
        runner = PolicyRunner(args.ckpt, device=dev, nfe=args.nfe, z0=args.z0)
        fmt = runner.format
        ck = Path(args.ckpt)
        step = int(ck.stem.split("_")[-1]) if ck.stem.startswith("ckpt_") else ""
        run_id = ck.parent.name
    note = args.note or ("wiring" if args.wiring else f"{run_id}_{step}")
    out = Path(args.out) if args.out else paths.REPO / "results" / f"{dt.date.today()}_policy_only_{note}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    outdir = Path(args.outdir) if args.outdir else out.with_suffix("")
    commit = paths.code_commit()
    new = not out.exists()
    f = open(out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=COLS + ["model_seconds", "state_err0", "image_err0", "command_mode", "point_error_cm",
                                       "min_dist_cmd_cm", "min_dist_true_cm", "demo_episode", "img_rowdiff0",
                                       "objective", "nfe", "z0", "determinism_fix"], extrasaction="ignore")
    if new:
        w.writeheader()
    cams = runner.cams if runner else ["left", "right"]
    for task in args.tasks:
        ds = paths.task_dir(task)
        H = args.horizon or paths.horizon(task)
        env = sim.make_env(ds, cams=sorted(set(cams) | {"left", "right"}), size=256,
                           depth=bool(runner and runner.format == "points_3d"))
        n_ok = 0
        for i in range(args.shard, args.episodes, args.nshards):
            seed = args.seed0 + i
            frames = [] if i < args.video else None
            torch.manual_seed(seed)              # policy noise is reproducible per episode seed
            extra_row = {}
            if args.wiring:
                # demo i: reset to its saved state, check the obs pipeline against the recording,
                # then play its actions through the same chunked loop
                obs0 = sim.reset_to_episode(env, ds, i)
                acts = sim.episode_actions(ds, i)
                from policy.data import load_task_table  # noqa: F401  (dataset state for comparison)
                import pandas as pd
                pq = next(ds.glob(f"data/*/episode_{i:06d}.parquet"))
                rec_state0 = np.asarray(pd.read_parquet(pq, columns=["observation.state"]).iloc[0, 0])
                extra_row["state_err0"] = float(np.abs(sim.state16(obs0) - rec_state0).max())
                from policy.extract_frames import decode_video, video_path
                v0 = decode_video(video_path(ds, "left", i))[0].astype(int)
                extra_row["image_err0"] = float(np.abs(sim.images(obs0, ["left"])[0].astype(int) - v0).mean())
                instr = sim.episode_extras(ds, i)[2].get("lang", "")
                r = run_episode(env, task, seed, lambda o, t: acts[t:t + 16], len(acts) + 50,
                                execute=args.execute, video=frames, reset_obs=obs0, record=args.save_traj)
            elif runner.format == "pose_path":
                # held-out demo scenes: episodes not in the run's training split
                tr = runner.cfg.get("episodes_json")
                from policy.extract_frames import episode_lengths
                n_eps = len(episode_lengths(ds))
                # no split recorded: the run trained on every demo, so every scene is a training scene
                train_eps = set(json.loads(Path(tr).read_text()).get(task, [])) if tr else set(range(n_eps))
                held = [e for e in range(n_eps) if e not in train_eps] or list(range(n_eps))
                ep = held[i % len(held)]
                obs0 = sim.reset_to_episode(env, ds, ep)
                from policy.data import load_task_table
                if not hasattr(runner, "_tables"):
                    runner._tables = {}
                if task not in runner._tables:
                    runner._tables[task] = load_task_table(task)
                tb = runner._tables[task]
                n = int(tb["lens"][ep]); g0 = int(tb["starts"][ep])
                from policy.label_commands import load_labels
                lbl = {l["episode"]: l for l in load_labels(task, paths.frames_base().parent / "labels")}[ep]
                t_ref = [0]
                runner.set_task(task, lbl["kv_text"], env=env, cmd_opts={"kv_text": lbl["kv_text"]})
                runner.set_demo(tb["states"][g0:g0 + n], t_ref)
                instr = lbl["kv_text"]

                def act_fn(o, t):
                    t_ref[0] = t
                    return runner.act(o)
                r = run_episode(env, task, seed, act_fn, H, execute=args.execute, video=frames,
                                reset_obs=obs0, record=args.save_traj)
                extra_row.update(command_mode=f"demo_scene_{ep}" + ("_TRAINED" if ep in train_eps else ""))
            else:
                if args.demo_scenes:
                    eps = json.loads(Path(args.demo_scenes).read_text())[task]
                    if i >= len(eps):
                        continue
                    obs0 = sim.reset_to_episode(env, ds, eps[i])
                    extra_row["demo_episode"] = eps[i]
                else:
                    env.rng = np.random.default_rng(seed)
                    obs0 = env.reset()
                instr = env.get_ep_meta().get("lang", "")
                wrong = args.distractor or (args.distractor_rate > 0 and
                                            np.random.default_rng(seed + 991).random() < args.distractor_rate)
                opts = dict(point_error_cm=args.point_error_cm, distractor=wrong, drop=args.drop,
                            rename=args.rename)
                cmd = runner.set_task(task, instr, env=env, cmd_opts=opts, rng=np.random.default_rng(seed + 7))
                instr = cmd["text"]
                track = {"cmd": [], "true": []}
                g_cmd, g_true = cmd.get("grasp_world"), cmd.get("true_grasp_world")

                def on_step(o):
                    e = np.asarray(o["robot0_eef_pos"])
                    if g_cmd is not None:
                        track["cmd"].append(np.linalg.norm(e - g_cmd))
                    if g_true is not None:
                        track["true"].append(np.linalg.norm(e - g_true))
                r = run_episode(env, task, seed, lambda o, t: runner.act(o), H, execute=args.execute,
                                video=frames, reset_obs=obs0, record=args.save_traj, on_step=on_step)
                mode = ("distractor" if wrong else "perfect") + (f"_rate{args.distractor_rate:g}" if args.distractor_rate else "") \
                    if (args.distractor or args.distractor_rate) else (f"drop_{args.drop}" if args.drop else
                                                             (f"rename" if args.rename else "perfect"))
                extra_row.update(command_mode=mode if runner.format != "sentence" else "sentence",
                                 point_error_cm=args.point_error_cm,
                                 min_dist_cmd_cm=round(100 * min(track["cmd"]), 1) if track["cmd"] else "",
                                 min_dist_true_cm=round(100 * min(track["true"]), 1) if track["true"] else "")
            # render sanity at reset: mean change between neighbouring pixel rows (about 6 clean, 25+ corrupt)
            extra_row["img_rowdiff0"] = round(float(np.abs(np.diff(sim.images(obs0, ["left"])[0].astype(np.int16),
                                                                   axis=0)).mean()), 1)
            if extra_row["img_rowdiff0"] > 20:
                print(f"WARNING corrupt-looking render at reset: rowdiff {extra_row['img_rowdiff0']}", flush=True)
            n_ok += r["success"]
            row = dict(task=task, episode=i, seed=seed, success=r["success"], seconds=r["seconds"], llm_calls=0,
                       input_tokens=0, output_tokens=0, cost_usd=0, code_commit=commit, checkpoint_step=step,
                       format=fmt, run_id=run_id or "wiring", steps=r["steps"], instruction=instr,
                       model_seconds=r["model_seconds"],
                       objective=getattr(runner, "objective", ""), nfe=getattr(runner, "nfe", ""),
                       z0=getattr(runner, "z0", ""),
                       determinism_fix=int(__import__("policy.determinism", fromlist=["x"]).enabled()), **extra_row)
            w.writerow(row)
            f.flush()
            print(json.dumps({k: row[k] for k in ("task", "episode", "success", "steps", "seconds", "instruction")}
                             | extra_row), flush=True)
            if frames:
                import imageio
                outdir.mkdir(parents=True, exist_ok=True)
                imageio.mimsave(outdir / f"{task}_ep{i:03d}_{'ok' if r['success'] else 'fail'}.mp4", frames, fps=20)
            if args.save_traj and "traj" in r:
                outdir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(outdir / f"{task}_ep{i:03d}_traj.npz", success=r["success"],
                                    final_state=env.sim.get_state().flatten(), **r["traj"])
        env.close()
        n_run = len(range(args.shard, args.episodes, args.nshards))
        print(f"EVAL {task}: {n_ok}/{n_run} succeeded", flush=True)
    f.close()
    print("wrote", out)


if __name__ == "__main__":
    main()
