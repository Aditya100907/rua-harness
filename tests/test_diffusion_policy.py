"""Diffusion Policy baseline: model wrapper, checkpoint round trip, training files. Skipped without the overlay."""
import json

import numpy as np
import pytest
import torch

from policy import diffusion_policy as dpm

try:
    dpm.import_lerobot_dp()
    HAVE = True
except ImportError:
    HAVE = False
pytestmark = pytest.mark.skipif(not HAVE, reason="LeRobot overlay not installed (setup/setup_dp_overlay.sh)")

TASKS = ["OpenDrawer", "TurnOnMicrowave"]


def tiny():
    return dpm.DPPolicy(TASKS, ["left", "wrist"], down_dims=(32, 64, 128), resize=64, infer_steps=3)


def fake_batch(B=3):
    return {"images": torch.randint(0, 255, (B, 2, 256, 256, 3), dtype=torch.uint8),
            "state": torch.rand(B, 16) * 2 - 1, "actions": torch.rand(B, 16, 12) * 2 - 1,
            "mask": torch.ones(B, 16, dtype=torch.bool), "task": torch.tensor([0, 1, 0][:B])}


def test_loss_and_predict_shapes():
    m = tiny()
    loss = m.loss(fake_batch())
    loss.backward()
    assert torch.isfinite(loss)
    m.eval()
    m.set_task("TurnOnMicrowave")
    a = m.predict(np.zeros((2, 256, 256, 3), np.uint8), np.zeros(16, np.float32))
    assert a.shape == (16, 12) and np.abs(a).max() <= 1.0 + 1e-5       # clipped to the sample range
    with pytest.raises(KeyError):
        m.set_task("CloseFridge")


def test_one_hot_changes_conditioning():
    m = tiny().eval()
    b = m._obs(torch.zeros(1, 2, 256, 256, 3, dtype=torch.uint8), torch.zeros(1, 16), torch.tensor([1]))
    assert b["observation.state"].shape == (1, 1, 18) and b["observation.state"][0, 0, 17] == 1


def test_save_load_round_trip(tmp_path):
    m = tiny().eval()
    p = tmp_path / "ckpt_0000010.pt"
    m.save(p)
    ck = torch.load(p, map_location="cpu", weights_only=True)      # PolicyRunner reads it this way
    assert ck["rua_format"] == "diffusion_policy"
    from policy.model import load_policy
    m2 = load_policy(p).eval()
    for (k, v), v2 in zip(m.state_dict().items(), m2.state_dict().values()):
        assert torch.equal(v, v2), k
    torch.manual_seed(0)
    a = m.predict(np.full((2, 256, 256, 3), 100, np.uint8), np.zeros(16, np.float32))
    torch.manual_seed(0)
    a2 = m2.predict(np.full((2, 256, 256, 3), 100, np.uint8), np.zeros(16, np.float32))
    assert np.allclose(a, a2)


def test_ema_moves_toward_weights():
    lin = torch.nn.Linear(2, 2)
    e = dpm.EMA(lin)
    with torch.no_grad():
        lin.weight.add_(1.0)
    for _ in range(50):
        e.update(lin)
    assert 0 < e.decay() < 1
    assert torch.allclose(e.model.weight, lin.weight, atol=0.2)


def test_train_dispatch_args():
    from policy import dp_train
    a = dp_train.parse_args(["--format", "diffusion_policy", "--all-tasks", "--out", "x"])
    assert len(a.tasks) == 18 and a.cams == ["left", "wrist"] and a.clip_norm == 1.0
