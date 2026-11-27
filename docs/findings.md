# Findings: current best understanding

Each claim, evidence for, evidence against, confidence. Revised as results arrive. Last revised Nov 27, 00:20.

| # | Claim | For | Against | Confidence |
|---|---|---|---|---|
| 1 | Our data pipeline and simulator conventions match the demos | replay 30/30, wiring 30/30, state at reset equal to 1e-8, image within 2 grey levels | none | high |
| 2 | Stock DEM, trained here, reproduces the paper's level (55.6% on 18 tasks) | none | run1 20k: 0% on drawer and pick-place; open-loop error worse than the mean action; output dominated by sampling noise, images barely used (journal Nov 27 06:45) | low: our training does not yet produce a working policy |
| 3 | Points help the policy more than names alone | GR00T 2 (2D points: 55.6% to 83.3% on short real tasks); X-Harness (sub-goal + 2D point 51.9% vs language 34.0%, no retraining) | X-Harness: VLM-predicted 3D poses 13 to 17% | low (no result of ours) |
| 4 | An agent with general tools already does well on RoboCasa | HexaShell: 59.3% without skills, 85.6% with skills on RoboCasa365 (their README; not run by us) | different split and seeds from ours | medium |
| 5 | Hindsight grasp points are a fair stand-in for an agent's points | labels within the object's extent on 120/120 checked pick demos | test-time "perfect" points are body origins, 2 to 6 cm median from the label points | medium |
