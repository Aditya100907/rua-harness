# Weekly update to Xiatao (DRAFT, not sent)

Facts as of the morning of Fri Nov 27, from `docs/MORNING_REPORT.md`, `docs/research_journal.md` and `docs/decisions_needed.md`. Adi to edit and send.

---

Hi Xiatao,

Happy Thanksgiving! A short update on the command-format project.

**Built and checked.** The data, training and evaluation pipeline for DEM on RoboCasa is done, and the checks we agreed on pass. Replaying recorded demo actions succeeds in 30 of 30 demos on the 3 smoke tasks, the same actions through our evaluation loop also succeed in 30 of 30, and an untrained checkpoint scores 0 of 28. Every demo of all 18 tasks now has a hindsight command label (grasp and place points from where the gripper closed and opened). Lifting a pixel to 3D with depth lands about 3 cm from the object. All five command formats are implemented and each trains on a small test. A first version of the agent side (an LLM with a "look" tool and a "run_skill" tool) ran one smoke-test episode end to end.

**Training now.** Stock DEM is training on the 3 smoke tasks (3 seeds, plus a left-plus-wrist camera run) and on all 18 tasks toward 300k steps to check we match the paper's 55.6%. That takes about 25 hours on one H100. The first format run (structured text) is training, and the point and pose-path runs are queued. I have no trained-policy success numbers yet.

**A question for you.** HexaShell's public repo now includes its own RoboCasa port and reports 59.3% without skills and 85.6% with skills on RoboCasa365 (from their README; we have not run it). Two things follow. Should William use their port as the agent-alone baseline instead of writing his own? And since a general-tools agent already does that well, should our claim rest more on time and tokens at similar success than on success alone? I'd value your take.

Thanks,
Adi
