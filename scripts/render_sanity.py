"""Check that renders stay clean across env resets once a DEM checkpoint is loaded.

Metric: mean absolute difference between neighbouring pixel rows of the left image (about 6 on clean
frames, 25 to 80 on corrupted ones, per the HexaShell-tool helper's observation on the Mac).
    python scripts/render_sanity.py CKPT [n_resets]
"""
import sys

import numpy as np

from policy import paths, sim


def rowdiff(img):
    return float(np.abs(np.diff(img.astype(np.int16), axis=0)).mean())


def main():
    ckpt = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    from policy.eval import PolicyRunner
    import torch
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    env = sim.make_env(paths.task_dir("OpenDrawer"), cams=["left", "right"])
    runner = PolicyRunner(ckpt, device=dev)
    for k in range(n):
        env.rng = np.random.default_rng(1000 + k)
        obs = env.reset()
        runner.set_task("OpenDrawer", env.get_ep_meta()["lang"], env=env)
        vals = [rowdiff(sim.images(obs, ["left"])[0])]
        for _ in range(3):
            chunk = runner.act(obs)
            for a in sim.lerobot_to_env_action(chunk[:8]):
                obs, _, _, _ = env.step(a)
            vals.append(rowdiff(sim.images(obs, ["left"])[0]))
        print(f"RESET {k}: rowdiff " + " ".join(f"{v:.1f}" for v in vals), flush=True)


if __name__ == "__main__":
    main()
