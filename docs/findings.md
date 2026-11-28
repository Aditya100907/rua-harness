# Findings: current best understanding

Each claim, evidence for, evidence against, confidence. Revised as results arrive. Last revised Nov 27, 00:20.

| # | Claim | For | Against | Confidence |
|---|---|---|---|---|
| 1 | Our data pipeline and simulator conventions match the demos | replay 30/30, wiring 30/30, state at reset equal to 1e-8, image within 2 grey levels | none | high |
| 2 | Stock DEM, trained here, reproduces the paper's level (55.6% on 18 tasks) | with regression instead of MeanFlow: 51.2% on 18 tasks at 300k (921/1800) | run1 20k: 0% on drawer and pick-place; open-loop error worse than the mean action; output dominated by sampling noise, images barely used (journal Nov 27 06:45) | medium-high with regression (one seed); MeanFlow as trained here does not learn |
| 3 | Points help the policy more than names alone | GR00T 2 (2D points 55.6% to 83.3%); X-Harness (sub-goal + 2D point 51.9% vs 34.0%); ours: points drawn on the images beat the sentence on drawer and microwave over 3 seeds (65 vs 53%, 56 vs 44%) | ours: no gain on pick-place; learned 3D point tokens are worse (1 seed); success is flat up to 10 cm of point error; the policy does not follow a point to a distractor (gripper near the distractor in 1 to 5 of 50) and collapses when the object name is removed, so it relies on the text | medium for 'drawn points raise success'; low for 'points steer the policy' |
| 4 | An agent with general tools already does well on RoboCasa | HexaShell: 59.3% without skills, 85.6% with skills on RoboCasa365 (their README; not run by us) | different split and seeds from ours | medium |
| 5 | Hindsight grasp points are a fair stand-in for an agent's points | labels within the object's extent on 120/120 checked pick demos | test-time "perfect" points are body origins, 2 to 6 cm median from the label points | medium |
