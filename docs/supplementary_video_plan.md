# Supplementary video plan (Part Z)

Source clips: `policy/eval.py` saves the first 2 episodes of every task and checkpoint as mp4 (left and right agent views side by side) under `$ROOT/runs/eval_media/<note>/` on Bouchet, named `<task>_epNNN_ok.mp4` or `_fail.mp4`. The commander saves its PNGs per call under `~/apollo-lab/runs/commander/<note>/`.

| # | Clip | Source | Status |
|---|---|---|---|
| 1 | Stock DEM success on each smoke task (drawer, microwave, pick-place) | run1_smoke final checkpoint, `_ok.mp4` | wait for 20k scoring (about 05:30) |
| 2 | A typical failure, by the most common failure type from E5 | run1_smoke, sorted with `--save-traj` | after E5 |
| 3 | The policy following a moved point: same scene, grasp point on the target vs on the distractor (N2) | points_3d_rgb final checkpoint, `distractor` sweep vs `err0cm` sweep, same seed | after F4 points runs (about 08:00) |
| 4 | Points drawn on the image (points_2d input as the policy sees it) | `docs/img/points_2d_check.png` style frames from the eval | after F4 points_2d |
| 5 | Commander episode: the agent looks, names a pixel, the policy runs | commander PNG sequence plus the agent's tool calls, stitched | after Part V runs |
| 6 | Scripted oracle vs learned policy on the same seed | `policy/scripted.py --video` and run1 eval, seed 1000 | anytime |

Rules: show the seed and checkpoint step on every clip; include at least one failure; no clip from a training scene presented as held-out.
