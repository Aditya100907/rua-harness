# Decisions log

Every decision made without asking Adi (per docs/STANDING_ORDERS.md), with its reason. Newest last.

- Fri Nov 27, 16:54. Objective: regression (mse) for the format study. Reason: gate at 5k: mse OpenDrawer 22/50 vs flow matching 0/50 on all three tasks; MeanFlow never beat the mean action.
- Fri Nov 27, 17:25. Determinism fix on by default in eval. Reason: same-seed scenes differed across processes (3 of 5 pick-place seeds); fix changed OpenDrawer by 1 episode.
- Fri Nov 27, 17:15. Stopped fs_fm_sentence to keep at 8 GPUs with the SAM 3 service up. Reason: fm lost the gate; its 5k checkpoint is kept.
- Fri Nov 27, 17:50. Stopped the idle SAM 3 service; run it on demand. Reason: YCRC cancels idle GPU jobs; no client was using it.
- Fri Nov 27, 19:40. Seed repeats for kv_text, points_2d, sentence. Reason: three seeds before a claim.
- Sat Nov 28, 13:36. Queued name dropout (points_2d, points_3d_rgb), held-out-object runs, 18-task points_2d and kv_text, second seeds of point-token formats. Reason: the distractor test showed the policy ignores points; name dropout tests whether points can carry the object choice (backlog 1).
- Sat Nov 28, 13:38. Adopted the standing orders.
- Sat Nov 28, 13:42. Adopted the research session's top recommendations (~/Desktop/APOLLO-Lab/research/litreview-2026-11-27/43_real_results_through_the_research.md): (1) twin-distractor training data for pick-place (identical mesh added to existing demos, re-rendered) so only the point identifies the target; (2) a referent axis in the tolerance sweep (wrong-object points on 10% and 30% of commands; 20 and 40 cm errors); (3) 3D points as distance channels in the point map instead of appended tokens. Reason: the distractor test shows the policy follows the name; in our training data the point never has to carry the referent, so formats can only differ in residual precision. These make the referent measurable. Priority: (1) is the critical path for backlog item 1.
