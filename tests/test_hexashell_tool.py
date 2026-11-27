"""run_skill as a HexaShell robot call (policy/hexashell_tool).

The live test calls the tool the way HexaShell's harness does: a cell calls `robot.run_skill(...)`
on a proxy whose `_call` pickles the arguments across (as kernel/worker.py does), the host's
`dispatch` answers, and the result object is pickled back. Errors come back as the exception
type HexaShell's worker raises for that kind.
"""
import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from policy.hexashell_tool import RunSkillHost, RunSkillProxy, Skill, ToolError, build_command, call_tool
from rua_harness.command import Command, Ref

RUNS = Path.home() / "apollo-lab/runs"
CKPT_SENTENCE = os.environ.get("RUA_TEST_CKPT", str(RUNS / "train_test/ckpt_0000050.pt"))
CKPT_POINTS = os.environ.get("RUA_TEST_CKPT_POINTS", str(RUNS / "f3_points_3d_rgb/ckpt_0000050.pt"))


class _Robot(RunSkillProxy):
    """HexaShell's worker side, reduced to the RPC: pickle in, dispatch, pickle out."""

    def __init__(self, host):
        self.host = host
        self.calls = []

    def _call(self, name, args, kwargs):
        name, args, kwargs = pickle.loads(pickle.dumps((name, args, kwargs)))
        self.calls.append((name, args, kwargs))
        try:
            out = self.host.dispatch(name, args, kwargs)
        except ToolError as exc:  # kernel/worker.py _rpc: error kind -> exception type
            raise {"ValueError": ValueError, "TypeError": TypeError}.get(exc.kind, RuntimeError)(exc.text) from None
        return pickle.loads(pickle.dumps(out))


def test_skill_text_and_command_dict():
    s = Skill(skill="open", words="skill open object drawer", reason="success_check", steps=12, seconds=0.6,
              points=[("object", (0.1, 0.2, 0.9), "left", (100, 120))], cams=["left", "right"], steps_left=700)
    text = str(s)
    assert text.startswith("Skill: open ran 12 steps (0.6 s)")
    assert "success" not in text.lower() and "[picture attached: left, right at the end]" in text
    cmd = Command("pick_place", object=Ref("apple"), target=Ref("cabinet"), source="counter")
    d = build_command(cmd, grasp=[1, 2, 3])
    assert d["kv_text"] == "skill pick place object apple from counter to cabinet"
    assert d["has"] == [True, False] and d["place_world"] is None and d["grasp_world"].dtype == np.float64


def _visible_handle_pixel(env, obs):
    """A pixel on the drawer handle seen by left or right: ground truth, used here only to play the agent."""
    from simlib.camera import views
    from simlib.truth import target_part
    part = target_part(env, "OpenDrawer")
    vs = views(env, obs, ["robot0_agentview_left", "robot0_agentview_right"])
    for cam, name in (("left", "robot0_agentview_left"), ("right", "robot0_agentview_right")):
        v = vs[name]
        for p in part.positions or [part.pos]:
            uv, z = v.world_to_pixel(p)
            u, w = int(round(uv[0])), int(round(uv[1]))
            if z > 0 and 0 <= u < 256 and 0 <= w < 256 and abs(v.depth[w, u] - z) < 0.03:
                return cam, (u, w), np.asarray(p)
    return None


def _frame_ok(img) -> bool:
    """Rendered frames, not renderer garbage. On the Mac, a hard reset after the first one followed by
    loading the model gives striped or noisy frames (mean row-to-row change 25 to 80; clean frames
    about 6), so this test resets once."""
    return float(np.abs(np.diff(np.asarray(img, int), axis=0)).mean()) < 20


@pytest.mark.sim
def test_run_skill_tool_live_opendrawer():
    if not (Path(CKPT_SENTENCE).exists() and Path(CKPT_POINTS).exists()):
        pytest.skip("test checkpoints missing")
    from policy import paths, sim
    env = sim.make_env(paths.task_dir("OpenDrawer"), cams=["left", "right"], depth=True)
    try:
        env.rng = np.random.default_rng(6)   # seed 6: the handle is in the left view at reset
        obs = env.reset()
        found = _visible_handle_pixel(env, obs)
        assert found, "drawer handle not in view at reset"
        cam, uv, handle = found

        # points checkpoint: the agent names a pixel on the handle; the harness lifts it with depth
        pics = []
        host = RunSkillHost(env, checkpoint=CKPT_POINTS, task="OpenDrawer", steps_left=lambda: 750,
                            attach=lambda kind, img, cap: pics.append((kind, img, cap)))
        res = call_tool(host, {"skill": "open", "object": {"name": "drawer", "camera": cam, "uv": list(uv)},
                               "max_seconds": 1.5})
        assert res["error"] is None, res["text"]
        assert res["text"].startswith("Skill: open ran") and "object point" in res["text"]
        assert len(res["images"]) == 2 and "success" not in res["text"].lower()
        assert [p[0] for p in pics] == ["run_skill", "run_skill"] and pics[0][1].shape == (256, 256, 3)
        assert all(_frame_ok(im) for _, im in res["images"]), "corrupt frames from the renderer"
        c = host.last["command"]
        assert host.last["format"] == "points_3d_rgb"
        assert c["kv_text"] == "skill open object drawer" and c["has"] == [True, False]
        assert np.linalg.norm(c["grasp_world"] - handle) < 0.04, (c["grasp_world"], handle)
        assert host.last["steps"] == 30 or host.last["reason"] == "success_check"

        # sentence checkpoint, same episode, through the HexaShell-style proxy: names only; a pixel is
        # accepted and reported unused
        logs = []
        host2 = RunSkillHost(env, checkpoint=CKPT_SENTENCE, task="OpenDrawer", log=logs.append,
                             steps_left=lambda: 720)
        robot = _Robot(host2)
        out = robot.run_skill("open", object={"name": "drawer", "camera": cam, "uv": list(uv)}, max_seconds=1.0)
        assert isinstance(out, Skill) and out.reason in ("max_seconds", "success_check")
        assert out.steps == 20 or out.reason == "success_check"
        assert host2.last["words"] == "Open the drawer." and host2.last["format"] == "sentence"
        assert any("words only" in n for n in out.notes) and out.cams == ["left", "right"]
        assert "success" not in str(out).lower() and not hasattr(out, "grading_only")
        assert "grading_only" in logs[-1]  # the record keeps it for scoring
        assert robot.calls[-1] == ("run_skill", ("open",), {"object": {"name": "drawer", "camera": cam,
                                   "uv": list(uv)}, "target": None, "source": None, "direction": None,
                                   "max_seconds": 1.0})

        # errors come back as HexaShell's exception kinds; nothing moves
        with pytest.raises(ValueError, match="needs field 'object'"):
            robot.run_skill("open")
        assert call_tool(host2, {"skill": "open", "object": {"name": "drawer", "camera": "top", "uv": [1, 2]}}
                         )["error"] == "ValueError"
    finally:
        env.close()
