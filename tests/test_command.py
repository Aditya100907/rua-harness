import json, numpy as np, pytest
from rua_harness import Command, ATOMIC_SEEN, SKILLS, to_policy_text
from rua_harness.command import to_points
from rua_harness.geometry import lift_pixel, depth_to_point_map, add_depth_noise


def test_all_18_tasks_validate_and_cover_all_skills():
    assert len(ATOMIC_SEEN) == 18
    cmds = [Command.from_json(json.dumps(c)) for c in ATOMIC_SEEN.values()]
    assert {c.skill for c in cmds} == set(SKILLS)


def test_text_has_no_numbers_or_punctuation():
    c = Command.from_json({"skill": "pick_place", "object": {"name": "apple", "point": [0.5, 0.1, 0.9]},
                           "source": "counter", "target": "cabinet"})
    assert to_policy_text(c) == "skill pick place object apple from counter to cabinet"
    pts, mask = to_points(c)
    assert pts[0] == [0.5, 0.1, 0.9] and mask == [True, False]


@pytest.mark.parametrize("bad", [
    {"skill": "fly"}, {"skill": "pick_place", "object": "apple"},
    {"skill": "open", "object": "drawer", "direction": "out"},
    {"skill": "turn", "object": "knob", "direction": "sideways"},
])
def test_bad_commands_rejected(bad):
    with pytest.raises((ValueError, TypeError)):
        Command.from_json(bad)


def test_lift_pixel_and_point_map_agree():
    K = np.array([[200.0, 0, 128], [0, 200.0, 128], [0, 0, 1]])
    T = np.eye(4); T[:3, 3] = [0.1, 0.2, 0.3]
    depth = np.full((256, 256), 2.0)
    p = lift_pixel([228, 128], depth, K, T)
    assert np.allclose(p, [1.1, 0.2, 2.3])          # 100 px right at 2 m with f=200 -> 1 m right
    assert np.allclose(depth_to_point_map(depth, K, T)[128, 228], p)


def test_depth_noise_makes_holes_and_keeps_shape():
    d = np.full((64, 64), 1.0); d[:, 32:] = 2.0
    n = add_depth_noise(d, np.random.default_rng(0))
    assert n.shape == d.shape and (n == 0).any() and abs(np.median(n[n > 0]) - 1.5) < 0.6
