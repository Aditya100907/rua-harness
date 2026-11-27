# Next experiments (Part L, step 4)

Ranked by information per GPU-hour. H100-hours unless noted (L40S is about 2.8x slower). Updated Nov 27, 00:30, before any trained-policy number exists.

| Rank | Question | Prediction | If yes / if no | Cost | Answer by |
|---|---|---|---|---|---|
| 1 | Does stock DEM on 3 tasks (run 1) score above 0 at 5k steps? (gate D3) | drawer and microwave above 20% by 10k | yes: the pipeline learns; no: check normalization, chunking, camera order (Part K) before anything else | running | 01:30 (5k), 05:00 (20k) |
| 2 | Does the learned policy beat the scripted baseline (F7: microwave 50/50, drawer about 50%, pick-place 40 to 70%)? | not at 20k on microwave; maybe on drawer | no: the learned part is not earning its place yet on these tasks; say so plainly | eval only | 05:30 |
| 3 | Do points help over text with perfect commands? (F4 points_3d_rgb vs kv_text vs run1) | points help pick-place by 10+ points, not drawer or microwave (instruction already decides) | yes: run the N1 error sweep; no: check the token reaches the head (Part K) and the noise scale | 3 runs x 1.7 h | 08:00 |
| 4 | Does the policy follow the point or the name? (N2 distractor on the points checkpoint) | follows the point more than half the time if points matter | decides whether 3D points can steer the policy at all | eval only | after 3 |
| 5 | How accurate must the agent be? (N1: 0, 1, 2, 5, 10 cm) | success holds to 2 cm, falls by 5 cm | sets the bar for Part V's agent error (commander smoke: 3.1 cm good camera, 81 cm occluded camera) | eval only | after 3 |
| 6 | Is left + wrist better than left + right? (run 5 vs run 1) | wrist helps pick-place | if yes, switch the format study's cameras before the 18-task runs | running | 05:00 |
| 7 | Seed noise (runs 3 and 4 vs run 1) | per-task spread under 10 points | sets which F4 differences count as real | running | 05:00 |
| 8 | Do points help on unseen objects? (F5 objsplit runs, held-out val) | the gap between points and sentence grows on held-out objects | the paper's generalization claim | 2 runs x 1.7 h | queued |
| 9 | Point map instead of color (points_3d vs points_3d_rgb) | similar on trained objects, better on held-out | the image-input question | 1.7 h, needs F2 depth | after F2 |
| 10 | Point-noise scale in training (0, 1, 3, 5 cm) | 2 to 3 cm best; 0 cm brittle to agent error | sets training noise for the 18-task runs | 4 x 1.7 h | after 3 and 5 |

Rules applied: each run changes one thing against run 1 (same seed, steps, batch); screening on 3 tasks before 18; a difference counts once it holds over 3 seeds or its interval clears zero.
