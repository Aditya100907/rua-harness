"""Open vision-language model as the commander: an agent loop with no Claude CLI.

    python -m policy.commander.open_model --ckpt CKPT --tasks OpenDrawer --episodes 1 --note smoke

Talks to an OpenAI-compatible endpoint (vLLM on Bouchet, policy/slurm/vlm_server.sbatch; the address is read
from runs/vlm_server.txt unless --url is given). Per episode:
  1. build the episode (policy.commander.server.Episode: env, policy, depth, grading-only target);
  2. show the model the left, right and wrist images and the job; it answers with one JSON object
     (vLLM structured output with JSON_SCHEMA below): run_skill with a command, or done;
  3. the command is validated, its pixels lifted to 3D with that camera's depth, and run_skill runs the
     policy (the harness converts the command to the checkpoint's format: sentence, kv_text, or points);
  4. new images go back to the model; repeat until it says done, robot time runs out, or --max-calls.
The model always gives names AND pixels, whatever the checkpoint's format, so the commander is the same
for every format and its pointing error is measured for every format (the sentence and kv_text policies
ignore the points).

Logged per model call (EPDIR/calls.jsonl): prompt and completion tokens, seconds, the raw reply.
Grading only (never shown to the model): success, distance of the lifted points to the true target.
Writes one CSV row per episode (results/README.md columns plus format, grasp_err_cm, place_err_cm, n_skill,
robot_seconds, ...). No paid API: the endpoint is our own server.
"""
from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import io
import json
import os
import time
import urllib.request
from pathlib import Path

import numpy as np

from policy import paths

ADDR_FILE = Path("/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/runs/vlm_server.txt")
CAMS = ("left", "right", "wrist")
IMG = 256   # simulator image size; pixels are lifted in this grid

COORDS = {
    # Qwen3-VL grounds in relative coordinates 0..1000 (its native convention)
    "norm1000": "uv is [x, y] in relative image coordinates from 0 to 1000: x from the left edge (0) to the "
                "right edge (1000), y from the top edge (0) to the bottom edge (1000).",
    # Qwen2.5-VL grounds in absolute pixels of the image it is given
    "pixel": "uv is [x, y] in pixels of the {s} x {s} image: x is the column from the left edge, y is the row "
             "from the top edge.",
}

SYSTEM = """You command a one-armed mobile robot in a simulated kitchen. A fast learned skill moves the arm; \
you decide what it should do and point at where it should act.

You see three cameras: "left" and "right" are fixed on the robot's base and look forward; "wrist" is on the gripper.

Reply with one JSON object and nothing else:
  {{"thought": "<one or two short sentences>", "action": "run_skill", "command": {{...}}}}   or
  {{"thought": "...", "action": "done"}}

The command:
  "skill": one of {skills}. Use pick_place to move an object somewhere, open / close for doors, drawers and lids,
      press for buttons and levers, turn for knobs and handles, slide for racks, insert to put one thing into another,
      navigate to drive to a place.
  "object": {{"name": short noun phrase, "camera": "left" | "right" | "wrist", "uv": [x, y]}}
      the thing the gripper touches (the object to pick, the handle or button to act on)
  "target": same form, where the object goes (pick_place, insert, navigate)
  "source": where the object starts, a name only (pick_place)
  "direction": one of {directions} (slide, turn, press)
  "max_seconds": robot seconds for this skill (optional; default all the remaining time)
{coords}
Pick each point on the thing itself, in a camera where it is clearly visible and not covered by the robot's arm.
The point is lifted to 3D with that camera's depth, so a point on the background is a wrong point.
Use the words of the job for the names.
After each skill you get new images. If the job is done, reply done. If not, you may send another command."""

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {"type": "string"},
        "action": {"type": "string", "enum": ["run_skill", "done"]},
        "command": {
            "type": "object",
            "properties": {
                "skill": {"type": "string", "enum": None},          # filled in from SKILLS
                "object": None,                                     # REF, filled in below
                "target": None,
                "source": {"type": "string"},
                "direction": {"type": "string", "enum": None},      # filled in from DIRECTIONS
                "max_seconds": {"type": "number"},
            },
            "required": ["skill"],
        },
    },
    "required": ["thought", "action"],
}
REF = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "camera": {"type": "string", "enum": list(CAMS)},
                   "uv": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}},
    "required": ["name", "camera", "uv"],
}


def schema() -> dict:
    from rua_harness.command import DIRECTIONS, SKILLS
    s = json.loads(json.dumps(JSON_SCHEMA))
    s["properties"]["command"]["properties"]["skill"]["enum"] = sorted(SKILLS)
    s["properties"]["command"]["properties"]["direction"]["enum"] = list(DIRECTIONS)
    s["properties"]["command"]["properties"]["object"] = dict(REF)
    s["properties"]["command"]["properties"]["target"] = dict(REF)
    return s


def server_url(url: str | None) -> str:
    if url:
        return url.rstrip("/")
    line = ADDR_FILE.read_text().splitlines()[0].strip()
    return f"http://{line}"


def png_b64(rgb: np.ndarray, size: int) -> str:
    from PIL import Image
    im = Image.fromarray(rgb)
    if size != rgb.shape[0]:
        im = im.resize((size, size), Image.BICUBIC)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def to_sim_pixel(uv, coords: str, sent: int):
    """Model coordinates -> integer pixel in the 256 x 256 simulator image, or None if outside."""
    x, y = float(uv[0]), float(uv[1])
    if coords == "norm1000":
        u, v = x / 1000 * IMG, y / 1000 * IMG
    else:
        u, v = x * IMG / sent, y * IMG / sent
    u, v = int(np.floor(u)), int(np.floor(v))
    u, v = min(u, IMG - 1) if u == IMG else u, min(v, IMG - 1) if v == IMG else v
    if not (0 <= u < IMG and 0 <= v < IMG):
        return None
    return [u, v]


class Chat:
    def __init__(self, url: str, model: str, max_tokens: int, timeout: int, log_path: Path):
        self.url, self.model, self.max_tokens, self.timeout = url, model, max_tokens, timeout
        self.log_path = log_path
        self.n, self.prompt_tokens, self.completion_tokens, self.seconds = 0, 0, 0, 0.0
        self.schema = schema()

    def __call__(self, messages: list) -> str:
        body = {"model": self.model, "messages": messages, "temperature": 0.0, "max_tokens": self.max_tokens,
                "seed": 0, "response_format": {"type": "json_schema",
                                               "json_schema": {"name": "command", "schema": self.schema}}}
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.loads(r.read())
        dt_s = time.time() - t0
        u = out.get("usage") or {}
        text = out["choices"][0]["message"]["content"] or ""
        self.n += 1
        self.prompt_tokens += int(u.get("prompt_tokens", 0))
        self.completion_tokens += int(u.get("completion_tokens", 0))
        self.seconds += dt_s
        with open(self.log_path, "a") as f:
            f.write(json.dumps({"call": self.n, "prompt_tokens": u.get("prompt_tokens"),
                                "completion_tokens": u.get("completion_tokens"), "seconds": round(dt_s, 2),
                                "finish_reason": out["choices"][0].get("finish_reason"), "reply": text}) + "\n")
        return text


def image_content(ep, size: int) -> list:
    parts = []
    for c in CAMS:
        parts.append({"type": "text", "text": f"{c} camera:"})
        parts.append({"type": "image_url",
                      "image_url": {"url": "data:image/png;base64," + png_b64(ep.views[c].rgb, size)}})
    return parts


def strip_old_images(messages: list, keep_last: int):
    """Replace images in all but the last `keep_last` user turns with a note (keeps the context short)."""
    users = [i for i, m in enumerate(messages) if m["role"] == "user" and isinstance(m["content"], list)]
    for i in users[:-keep_last] if keep_last else users:
        messages[i]["content"] = [p if p["type"] == "text" else {"type": "text", "text": "(image omitted)"}
                                  for p in messages[i]["content"]]


def check_command(cmd: dict, coords: str, sent: int) -> tuple[dict | None, str | None]:
    """Model command -> server command (pixels in the simulator grid), or an error message for the model."""
    from rua_harness.command import SKILLS
    if not isinstance(cmd, dict) or "skill" not in cmd:
        return None, "the command needs a skill"
    if cmd["skill"] not in SKILLS:
        return None, f"unknown skill {cmd['skill']!r}"
    out = {"skill": cmd["skill"]}
    for k in ("source", "direction", "max_seconds"):
        if cmd.get(k) not in (None, ""):
            out[k] = cmd[k]
    for k in ("object", "target"):
        r = cmd.get(k)
        if not r:
            continue
        uv = to_sim_pixel(r["uv"], coords, sent)
        if uv is None:
            return None, f"{k}.uv {r['uv']} is outside the image"
        out[k] = {"name": r["name"], "camera": r["camera"], "uv": uv, "uv_model": list(r["uv"])}
    required, optional = SKILLS[cmd["skill"]]
    for f in required:
        if f not in out:
            return None, f"skill {cmd['skill']!r} needs field {f!r}"
    for f in ("object", "target", "source", "direction"):
        if f in out and f not in required + optional:
            out.pop(f)       # drop fields the skill does not take (validation would refuse them)
    return out, None


def run_episode(task: str, seed: int, ckpt: str, epdir: Path, chat_args: dict, coords: str, img_size: int,
                max_calls: int, wall_limit: float, keep_images: int, snap: bool = False) -> dict:
    from policy.commander.server import Episode
    from rua_harness.command import DIRECTIONS, SKILLS
    epdir.mkdir(parents=True, exist_ok=True)
    for f in ("calls.jsonl",):
        (epdir / f).unlink(missing_ok=True)
    chat = Chat(log_path=epdir / "calls.jsonl", **chat_args)
    ep = Episode(task, seed, ckpt, epdir, snap=snap)
    t0 = time.time()
    ep.look({})
    sysmsg = SYSTEM.format(skills=", ".join(sorted(SKILLS)), directions=", ".join(DIRECTIONS),
                           coords=COORDS[coords].format(s=img_size))
    messages = [{"role": "system", "content": sysmsg},
                {"role": "user", "content": [{"type": "text", "text": f"Job: {ep.lang}\nRobot time: "
                                              f"{ep.horizon / 20:.0f} seconds in total. Current images:"}]
                 + image_content(ep, img_size)}]
    status, commands = "max_calls", []
    while chat.n < max_calls:
        if time.time() - t0 > wall_limit:
            status = "wall_limit"
            break
        try:
            text = chat(messages)
        except Exception as e:   # server down or timeout: the episode ends, recorded as such
            status = f"llm_error: {type(e).__name__}: {e}"[:200]
            break
        messages.append({"role": "assistant", "content": text})
        try:
            reply = json.loads(text)
        except json.JSONDecodeError:
            messages.append({"role": "user", "content": "That was not valid JSON. Reply with one JSON object."})
            continue
        if reply.get("action") == "done":
            status = "done"
            break
        cmd, err = check_command(reply.get("command"), coords, img_size)
        if err:
            messages.append({"role": "user", "content": f"Command refused: {err}. Send a corrected command."})
            continue
        left_s = (ep.horizon - ep.steps) / 20
        cmd.setdefault("max_seconds", left_s)
        r = ep.run_skill({k: v for k, v in cmd.items()})
        commands.append({"command": cmd, "result": {k: v for k, v in r.items() if k != "images"}})
        if "error" in r:
            messages.append({"role": "user", "content": f"Command refused: {r['error']}. Send a corrected command."})
            continue
        strip_old_images(messages, keep_images - 1 if keep_images > 0 else 0)
        if r.get("status") == "out of time" or ep.steps >= ep.horizon:
            status = "out_of_time"
            break
        messages.append({"role": "user", "content": [{"type": "text", "text":
                         f"The skill stopped after {r['robot_seconds']:.1f} robot seconds. Robot time left: "
                         f"{r['robot_seconds_left']:.1f} seconds. New images:"}] + image_content(ep, img_size)})
    ep.finish({})
    fin = ep.final({})
    (epdir / "final.json").write_text(json.dumps({**fin, "commands": commands, "status": status}, indent=1,
                                                 default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    ep.env.close()
    g = [l["grasp_err_cm"] for l in fin["log"] if "grasp_err_cm" in l]
    p = [l["place_err_cm"] for l in fin["log"] if "place_err_cm" in l]
    return {"success": fin["success"], "seconds": round(time.time() - t0, 1), "llm_calls": chat.n,
            "input_tokens": chat.prompt_tokens, "output_tokens": chat.completion_tokens, "cost_usd": 0,
            "llm_seconds": round(chat.seconds, 1), "robot_seconds": fin["robot_seconds"], "n_skill": fin["n_skill"],
            "grasp_err_cm": g[0] if g else "", "place_err_cm": p[0] if p else "",
            "grasp_err_cm_all": ";".join(map(str, g)), "place_err_cm_all": ";".join(map(str, p)),
            "agent_status": status, "instruction": fin["lang"], "snap": int(snap),
            "snapped_to_target": ";".join(str(l.get("snapped_to_target")) for l in fin["log"]),
            "sent_text": " | ".join(l["sent_text"] for l in fin["log"]),
            "sentence": " | ".join(l["sentence"] for l in fin["log"])}


COLS = ["task", "episode", "seed", "success", "seconds", "llm_calls", "input_tokens", "output_tokens", "cost_usd",
        "code_commit", "checkpoint_step", "format", "run_id", "robot_seconds", "n_skill", "grasp_err_cm",
        "place_err_cm", "llm_seconds", "grasp_err_cm_all", "place_err_cm_all", "agent_status", "model", "coords",
        "img_size", "instruction", "sent_text", "sentence", "snap", "snapped_to_target"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tasks", nargs="+", default=paths.SMOKE_TASKS)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=1000, help="episode i uses seed seed0 + i (as policy/eval.py)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--note", default="smoke")
    ap.add_argument("--out", default=None, help="CSV (default results/<date>_agent_plus_policy_vlm_<note>.csv)")
    ap.add_argument("--epdir", default=None, help="episode folders (default runs/vlm_commander/<note>)")
    ap.add_argument("--url", default=None, help="server URL (default: read runs/vlm_server.txt)")
    ap.add_argument("--model", default="vlm", help="served model name")
    ap.add_argument("--model-id", default=os.environ.get("VLM_MODEL_ID", ""), help="for the CSV")
    ap.add_argument("--coords", default="norm1000", choices=list(COORDS))
    ap.add_argument("--img-size", type=int, default=512, help="images are upscaled to this before sending")
    ap.add_argument("--max-calls", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--keep-images", type=int, default=2, help="image sets kept in the context")
    ap.add_argument("--wall-limit", type=float, default=1800)
    ap.add_argument("--snap", action="store_true",
                    help="decode-first control: snap the agent's grasp point to the nearest plausible referent")
    args = ap.parse_args()
    ck = Path(args.ckpt)
    fmt = json.loads((ck.parent / "config.json").read_text()).get("format", "sentence")
    out = Path(args.out) if args.out else paths.REPO / "results" / f"{dt.date.today()}_agent_plus_policy_vlm_{args.note}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    root = Path(args.epdir) if args.epdir else paths.REPO.parent / "runs" / "vlm_commander" / args.note
    url = server_url(args.url)
    chat_args = dict(url=url, model=args.model, max_tokens=args.max_tokens, timeout=600)
    new = not out.exists()
    f = open(out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
    if new:
        w.writeheader()
    commit = paths.code_commit()
    for task in args.tasks:
        for i in range(args.shard, args.episodes, args.nshards):
            seed = args.seed0 + i
            import torch
            torch.manual_seed(seed)
            r = run_episode(task, seed, str(ck), root / f"{task}_{seed}", chat_args, args.coords, args.img_size,
                            args.max_calls, args.wall_limit, args.keep_images, snap=args.snap)
            row = dict(task=task, episode=i, seed=seed, code_commit=commit, format=fmt,
                       checkpoint_step=ck.stem.split("_")[-1].lstrip("0") or "0", run_id=ck.parent.name,
                       model=args.model_id or args.model, coords=args.coords, img_size=args.img_size, **r)
            w.writerow(row)
            f.flush()
            print(json.dumps({k: row[k] for k in ("task", "seed", "success", "seconds", "llm_calls", "input_tokens",
                                                   "output_tokens", "n_skill", "grasp_err_cm", "place_err_cm",
                                                   "agent_status", "sent_text")}), flush=True)
    f.close()
    print("wrote", out)


if __name__ == "__main__":
    main()
