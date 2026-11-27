"""Holds one episode. JSON over HTTP on localhost. The agent reaches it only through policy/commander/cli.py.

    python -m policy.commander.server --task OpenDrawer --seed 1000 --ckpt CKPT --dir EPDIR --port 8765

Endpoints (POST, JSON):
  /look       -> saves left, right, wrist PNGs in EPDIR/img, returns their paths
  /run_skill  -> a command in rua_harness.command.TOOL_SCHEMA form; pixels are lifted to 3D with depth
                 here; runs policy.run_skill; returns status and new image paths (never ground truth)
  /done       -> the agent says it is finished
  /final      -> grading only (called by the runner, not the CLI): success, robot seconds, point errors
Robot time is capped at the task's horizon; when it runs out, run_skill returns "out of time".
"""
from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

from policy import paths, sim

CAM_NAMES = {"left": "robot0_agentview_left", "right": "robot0_agentview_right", "wrist": "robot0_eye_in_hand"}


class Episode:
    def __init__(self, task, seed, ckpt, epdir: Path, size=256):
        from simlib.truth import target_part
        self.task, self.seed, self.ckpt, self.dir = task, seed, ckpt, epdir
        (epdir / "img").mkdir(parents=True, exist_ok=True)
        self.env = sim.make_env(paths.task_dir(task), cams=["left", "right", "wrist"], size=size, depth=True)
        self.env.rng = np.random.default_rng(seed)
        self.obs = self.env.reset()
        self.lang = self.env.get_ep_meta().get("lang", "")
        self.horizon = paths.horizon(task)
        self.steps, self.n_look, self.n_skill, self.success, self.done = 0, 0, 0, False, False
        self.log = []
        self.part = target_part(self.env, task)        # grading only
        self.t0 = time.time()

    def _save_images(self) -> dict:
        import imageio
        from simlib.camera import views
        v = views(self.env, self.obs, [CAM_NAMES[c] for c in ("left", "right", "wrist")])
        self.views = {c: v[CAM_NAMES[c]] for c in ("left", "right", "wrist")}
        out = {}
        k = self.n_look + self.n_skill
        for c, view in self.views.items():
            p = self.dir / "img" / f"{k:03d}_{c}.png"
            imageio.imwrite(p, view.rgb)
            out[c] = str(p)
        return out

    def look(self, _):
        self.n_look += 1
        return {"images": self._save_images(), "image_size": [256, 256],
                "pixel_convention": "uv = [column, row], origin top-left", "robot_seconds_used": self.steps / 20}

    def _lift(self, ref):
        if not ref or "uv" not in ref or "camera" not in ref:
            return None
        if not hasattr(self, "views"):
            self._save_images()
        view = self.views[ref["camera"]]
        u, v = int(ref["uv"][0]), int(ref["uv"][1])
        if not (0 <= u < 256 and 0 <= v < 256):
            return None
        return np.asarray(view.pixel_to_world([u, v]), dtype=np.float64).reshape(3)

    def run_skill(self, cmd):
        from policy.run_skill import run_skill
        from rua_harness.command import Command, Ref, to_policy_text, to_sentence, validate
        if self.done:
            return {"error": "episode is over"}
        left = self.horizon - self.steps
        if left <= 0:
            return {"status": "out of time", "robot_seconds_used": self.steps / 20}
        try:
            c = Command(skill=cmd["skill"],
                        object=Ref(cmd["object"]["name"]) if cmd.get("object") else None,
                        target=Ref(cmd["target"]["name"]) if cmd.get("target") else None,
                        source=cmd.get("source"), direction=cmd.get("direction"),
                        max_seconds=float(cmd.get("max_seconds", 30)))
            validate(c)
        except Exception as e:
            return {"error": f"bad command: {e}"}
        g = self._lift(cmd.get("object"))
        p = self._lift(cmd.get("target"))
        command = {"kv_text": to_policy_text(c), "grasp_world": g, "place_world": p,
                   "has": [g is not None, p is not None]}
        sentence = to_sentence(c)
        self.n_skill += 1
        r = run_skill(self.env, sentence, max_seconds=min(c.max_seconds, left / 20), checkpoint=self.ckpt,
                      task=self.task, command=command, obs=self.obs)
        self.obs = r["obs"]
        self.steps += r["steps"]
        self.success = self.success or r["grading_only"]["success"] or bool(self.env._check_success())
        # grading: distance of the lifted points to the true target (logged, never returned)
        err = {}
        if g is not None:
            ref = self.part.object_pos if self.part.object_pos is not None else self.part.pos
            err["grasp_err_cm"] = round(100 * float(np.linalg.norm(g - np.asarray(ref))), 1)
        if p is not None and self.part.dest_pos is not None:
            err["place_err_cm"] = round(100 * float(np.linalg.norm(p - np.asarray(self.part.dest_pos))), 1)
        self.log.append({"command": cmd, "sent_text": command["kv_text"], "sentence": sentence, "steps": r["steps"],
                         "reason": r["reason"], **err})
        imgs = self._save_images()
        return {"status": "skill stopped", "robot_seconds": r["robot_seconds"],
                "robot_seconds_used": self.steps / 20, "robot_seconds_left": (self.horizon - self.steps) / 20,
                "images": imgs}

    def finish(self, _):
        self.done = True
        return {"ok": True}

    def final(self, _):
        errs = [l["grasp_err_cm"] for l in self.log if "grasp_err_cm" in l]
        perrs = [l["place_err_cm"] for l in self.log if "place_err_cm" in l]
        return {"success": int(self.success), "robot_seconds": self.steps / 20, "n_look": self.n_look,
                "n_skill": self.n_skill, "grasp_err_cm_first": errs[0] if errs else None,
                "place_err_cm_first": perrs[0] if perrs else None, "log": self.log, "lang": self.lang,
                "wall_seconds": round(time.time() - self.t0, 1)}


def serve(ep: Episode, port: int):
    routes = {"/look": ep.look, "/run_skill": ep.run_skill, "/done": ep.finish, "/final": ep.final}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            fn = routes.get(self.path)
            try:
                out = fn(body) if fn else {"error": "unknown endpoint"}
            except Exception as e:      # report, keep serving
                out = {"error": f"{type(e).__name__}: {e}"}
            data = json.dumps(out, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", port), H)
    (ep.dir / "READY").write_text(str(port))
    srv.serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    serve(Episode(a.task, a.seed, a.ckpt, Path(a.dir)), a.port)


if __name__ == "__main__":
    main()
