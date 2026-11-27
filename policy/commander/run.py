"""Run commander episodes: one episode server plus one headless coding-agent run per episode.

    python -m policy.commander.run --ckpt CKPT --tasks OpenDrawer --episodes 2 --note smoke

Uses the `claude` CLI in print mode on the logged-in subscription. ANTHROPIC_API_KEY and
similar variables are removed from the agent's environment so no API billing can happen.
The agent may only run ./rua (the two tools) and read files. Tokens, turns and seconds come
from the CLI's JSON result; cost_usd is the CLI's list-price figure (not money spent).
Writes results/<date>_agent_plus_policy_<note>.csv.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from policy import paths

PROMPT = """You control a one-armed mobile robot in a simulated kitchen. Your job: {lang}

You have exactly two tools, run from the shell in this folder:
  ./rua look
      Saves the robot's current camera images (left, right, wrist; 256 x 256 PNG) and prints their paths.
      Open them with the Read tool to see the scene.
  ./rua run_skill '<json command>'
      Hands one job to the robot's learned skill and waits until the skill stops. Prints the new image paths.
      The command (JSON):
        "skill": one of {skills}
        "object": {{"name": short noun phrase, "camera": "left"|"right"|"wrist", "uv": [column, row]}}
             the thing the gripper touches; uv is the pixel (origin top-left) of where to act on it,
             e.g. the grasp point on an object or the handle of a door
        "target": same form, where it goes (for pick_place / insert / navigate)
        "source": where the object starts, a name only (pick_place)
        "direction": one of on, off, in, out, up, down, left, right (slide / turn / press)
        "max_seconds": robot seconds to allow (default 30)
      Example: ./rua run_skill '{{"skill": "pick_place", "object": {{"name": "apple", "camera": "right", "uv": [120, 140]}}, "source": "counter", "target": {{"name": "cabinet", "camera": "left", "uv": [130, 40]}}}}'
  ./rua done
      Say you are finished.

Pick each pixel on the object itself, where nothing (such as the robot's own arm) covers it; the point is
lifted to 3D from that camera's depth. Look first. Then send commands. After each skill, look at the new images and decide whether the job is done;
if not, you may send another command. The robot has {seconds:.0f} seconds of robot time in total.
Do not write or edit files. When finished, run ./rua done and reply with one line.
"""
FORBIDDEN_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_API_KEY", "OPENAI_API_KEY")


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def post(port: int, path: str, body=None):
    import urllib.request
    req = urllib.request.Request(f"http://127.0.0.1:{port}/{path}", data=json.dumps(body or {}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def run_one(task, seed, ckpt, epdir: Path, timeout_s: int, model: str | None) -> dict:
    from rua_harness.command import SKILLS
    epdir.mkdir(parents=True, exist_ok=True)
    port = free_port()
    py = sys.executable
    env_srv = dict(os.environ, PYTHONPATH=str(paths.REPO), PYTORCH_ENABLE_MPS_FALLBACK="1")
    srv = subprocess.Popen([py, "-m", "policy.commander.server", "--task", task, "--seed", str(seed), "--ckpt",
                            ckpt, "--dir", str(epdir), "--port", str(port)], cwd=paths.REPO, env=env_srv,
                           stdout=open(epdir / "server.log", "w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(600):
            if (epdir / "READY").exists() or srv.poll() is not None:
                break
            time.sleep(0.5)
        if not (epdir / "READY").exists():
            raise RuntimeError(f"episode server did not start; see {epdir / 'server.log'}")
        rua = epdir / "rua"
        rua.write_text(f"#!/bin/bash\nRUA_CMD_PORT={port} PYTHONPATH={paths.REPO} exec {py} -m policy.commander.cli \"$@\"\n")
        rua.chmod(0o755)
        lang = json.loads(json.dumps(post(port, "final")))["lang"]
        prompt = PROMPT.format(lang=lang, skills=", ".join(sorted(SKILLS)), seconds=paths.horizon(task) / 20)
        env = {k: v for k, v in os.environ.items() if k not in FORBIDDEN_ENV}
        cmd = ["claude", "-p", prompt, "--output-format", "json", "--no-session-persistence",
               "--allowedTools", "Bash(./rua:*)", "Read",
               "--disallowedTools", "Edit", "Write", "WebFetch", "WebSearch", "NotebookEdit"]
        if model:
            cmd += ["--model", model]
        t0 = time.time()
        try:
            p = subprocess.run(cmd, cwd=epdir, env=env, capture_output=True, text=True, timeout=timeout_s)
            out = p.stdout
            status = "ok" if p.returncode == 0 else f"exit {p.returncode}"
        except subprocess.TimeoutExpired as e:
            out, status = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""), "timeout"
        wall = time.time() - t0
        (epdir / "agent_result.json").write_text(out or "")
        try:
            res = json.loads(out)
        except Exception:
            res = {}
        fin = post(port, "final")
        (epdir / "final.json").write_text(json.dumps(fin, indent=1))
        u = res.get("usage", {}) or {}
        return {"success": fin["success"], "seconds": round(wall, 1), "robot_seconds": fin["robot_seconds"],
                "llm_calls": res.get("num_turns", ""),
                "input_tokens": (u.get("input_tokens", 0) or 0) + (u.get("cache_read_input_tokens", 0) or 0)
                + (u.get("cache_creation_input_tokens", 0) or 0),
                "output_tokens": u.get("output_tokens", ""), "cost_usd": res.get("total_cost_usd", ""),
                "n_look": fin["n_look"], "n_skill": fin["n_skill"], "grasp_err_cm": fin["grasp_err_cm_first"],
                "place_err_cm": fin["place_err_cm_first"], "agent_status": status,
                "model": ",".join((res.get("modelUsage") or {}).keys()), "instruction": fin["lang"]}
    finally:
        srv.terminate()
        try:
            srv.wait(10)
        except Exception:
            srv.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tasks", nargs="+", default=paths.SMOKE_TASKS)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--note", default="commander")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--model", default=None)
    ap.add_argument("--budget", type=int, default=60, help="hard cap on agent episodes in this call")
    args = ap.parse_args()
    if shutil.which("claude") is None:
        sys.exit("claude CLI not found")
    ck = Path(args.ckpt)
    fmt = json.loads((ck.parent / "config.json").read_text()).get("format", "sentence")
    out = paths.REPO / "results" / f"{dt.date.today()}_agent_plus_policy_{args.note}.csv"
    cols = ["task", "episode", "seed", "success", "seconds", "llm_calls", "input_tokens", "output_tokens",
            "cost_usd", "code_commit", "checkpoint_step", "format", "run_id", "robot_seconds", "n_look", "n_skill",
            "grasp_err_cm", "place_err_cm", "agent_status", "model", "instruction"]
    new = not out.exists()
    f = open(out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    if new:
        w.writeheader()
    n = 0
    for task in args.tasks:
        for i in range(args.episodes):
            if n >= args.budget:
                break
            seed = args.seed0 + i
            epdir = Path.home() / "apollo-lab" / "runs" / "commander" / args.note / f"{task}_{seed}"
            r = run_one(task, seed, str(ck), epdir, args.timeout, args.model)
            n += 1
            row = dict(task=task, episode=i, seed=seed, code_commit=paths.code_commit(), format=fmt,
                       checkpoint_step=ck.stem.split("_")[-1].lstrip("0") or "0", run_id=ck.parent.name, **r)
            w.writerow(row)
            f.flush()
            print(json.dumps({k: row[k] for k in ("task", "seed", "success", "seconds", "llm_calls", "input_tokens",
                                                   "output_tokens", "n_skill", "grasp_err_cm", "agent_status")}),
                  flush=True)
    f.close()
    print("wrote", out, "episodes run:", n)


if __name__ == "__main__":
    main()
