"""Checks for policy/testsets.py and policy/scripted.py that need no simulator."""
import json

import numpy as np
import pytest

from policy import paths, sim, testsets
from policy.scripted import env_to_lerobot_action, rot_from_axes


def test_action_order_roundtrip():
    a = np.arange(12, dtype=float)
    assert np.allclose(sim.lerobot_to_env_action(env_to_lerobot_action(a), binarize=False), a)


def test_rot_from_axes_is_rotation():
    R = rot_from_axes([1, 0, -1], [0, 1, 0])
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(R[:, 2], np.array([1, 0, -1]) / np.sqrt(2))


def test_kitchen_sets_avoid_target_pairs():
    target = {(i, i) for i in range(1, 11)}
    assert not target & set(testsets.KITCHENS_RECOMBINED)
    assert not target & set(testsets.KITCHENS_PRETRAIN)
    assert all(l > 10 and s > 10 for l, s in testsets.KITCHENS_PRETRAIN)


def _train_instructions(task):
    try:
        ds = paths.task_dir(task)
    except FileNotFoundError:
        pytest.skip(f"no {task} dataset here")
    with open(ds / "meta" / "tasks.jsonl") as f:
        return [json.loads(l)["task"].lower() for l in f]


def test_unseen_names_never_in_training_instructions():
    instr = _train_instructions("PickPlaceCounterToCabinet")
    cats = set(testsets.UNSEEN_NAMES)
    for cat, alts in testsets.UNSEEN_NAMES.items():
        for alt in alts:
            assert alt not in cats, (cat, alt)
            assert not any(f" {alt} " in f" {s} " for s in instr), (cat, alt)


def test_unseen_phrases_never_in_training_instructions():
    for task, table in testsets.UNSEEN_PHRASES.items():
        instr = set(_train_instructions(task))
        for orig, alts in table.items():
            assert orig.lower() in instr
            assert not any(a.lower() in instr for a in alts)


def test_unseen_names_cover_every_training_category():
    import re
    cats = {m.group(1) for s in _train_instructions("PickPlaceCounterToCabinet")
            if (m := re.match(r"pick the (.*) from the counter", s))}
    assert cats and cats <= {c.lower() for c in testsets.UNSEEN_NAMES}
