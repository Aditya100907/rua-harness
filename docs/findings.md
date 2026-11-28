# Findings: current best understanding

Each claim, evidence for, evidence against, confidence. Revised as results arrive. Last revised Nov 27, 00:20.

| # | Claim | For | Against | Confidence |
|---|---|---|---|---|
| 1 | Our data pipeline and simulator conventions match the demos | replay 30/30, wiring 30/30, state at reset equal to 1e-8, image within 2 grey levels | none | high |
| 2 | Stock DEM, trained here, reproduces the paper's level (55.6% on 18 tasks) | with regression instead of MeanFlow: 51.2% on 18 tasks at 300k (921/1800) | run1 20k: 0% on drawer and pick-place; open-loop error worse than the mean action; output dominated by sampling noise, images barely used (journal Nov 27 06:45) | medium-high with regression (one seed); MeanFlow as trained here does not learn |
| 3 | Points help the policy more than names alone | GR00T 2 (2D points 55.6% to 83.3%); X-Harness (sub-goal + 2D point 51.9% vs 34.0%); ours: points drawn on the images beat the sentence on drawer and microwave over 3 seeds (65 vs 53%, 56 vs 44%) | ours: no gain on pick-place; learned 3D point tokens are worse (1 seed); success is flat up to 10 cm of point error; the policy does not follow a point to a distractor (gripper near the distractor in 1 to 5 of 50) and collapses when the object name is removed, so it relies on the text | medium for 'drawn points raise success'; low for 'points steer the policy' |
| 4 | An agent with general tools already does well on RoboCasa | HexaShell: 59.3% without skills, 85.6% with skills on RoboCasa365 (their README; not run by us) | different split and seeds from ours | medium |
| 5 | Hindsight grasp points are a fair stand-in for an agent's points | labels within the object's extent on 120/120 checked pick demos | test-time "perfect" points are body origins, 2 to 6 cm median from the label points | medium |

## Dec 13 decision number (from the research synthesis; tracked from Nov 28)
**Definition:** the perfect-command success gap between the best and worst of F2 (kv_text), F3 (points_2d) and F4 (3D points + point map) at ambiguity level L3 (identical-twin scenes), counted only if every format loses at least 15 points without its command (no-command control). Gap of 10+ points: continue to Jan 22. Below: move the work to NeurIPS (Adi decides).
| Date | Gap at L3 | No-command drops | Status |
|---|---|---|---|
| Nov 28 | not measurable yet: no L3 (twin) scenes or twin-trained checkpoints; at L0 the F2/F3/F4 spread on 3 tasks is points_2d 65 / kv_text 55 / points_3d 41 on the drawer (old labels, points_3d one seed) | no-command control running (first results due today) | twin data being built; relabelled runs next |

| 6 | Under a real agent, a coordinate format loses its perfect-command lead | pilot (Qwen3-VL-8B, 10 episodes per task, old labels): points_2d 17/30 perfect vs 6/30 agent; sentence 10/30 vs 5/30; agent pointing error 10 to 16 cm, at or beyond the policy's capture radius | one seed, small n, old labels | low (pilot) |
