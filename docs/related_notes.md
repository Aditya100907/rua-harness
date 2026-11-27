# Related work notes

Reading notes for the command format question: what should a high-level agent send a low-level policy, and is the policy retrained for each format? Written 2026-11-26 (task E6).

Each entry says how much was read. "Read in full" means the main text was read end to end. "Skimmed" means sections were found by search and read in part. "Abstract only" means nothing past the abstract.

## X-Harness (arXiv 2611.34778)

Lai, Zhong, Li, et al. "Toward Agentic Robotics: A Large-Scale Study of Harness Design for Robot-Use Agents." https://arxiv.org/abs/2611.34778. Code and logs: github.com/google-research/x-as-policies.

**What was read.** Read in full: abstract and main text, sections 1 to 7 (from the arXiv HTML version). Read: Appendix D (models), the Table 4 checkpoint list and Appendix F.1 (action level numbers). Skimmed by keyword search: the rest of the appendices. Caveat: the HTML-to-text step dropped inline math, so a few numbers in the main text were lost. The numbers below come from tables that came through intact.

**What they test.** They define the "policy harness" as everything between the VLM (vision-language model) and the low-level policy: the context the VLM gets, its tools, whether it writes policy code or acts in the loop, and when it is called. They run over 470,000 rollouts on 8 simulated benchmarks (RoboCasa365 is one), with 6 VLMs and 15 VLA (vision-language-action) or WAM (world-action model) checkpoints. Study 1 is the part closest to our question. It compares what the VLM outputs: raw 3D gripper poses, key poses plus a motion planner, language sub-goals to a VLA, and language sub-goals plus a 2D target point marked on the image. On RoboMME with Gemini Flash (Table 6a): raw poses 13.1%, key poses 17.3%, language sub-goals 34.0%, sub-goals plus target point 51.9%, VLA alone 46.5%. With oracle sub-goals from the simulator, target-marked reaches 83.1% against 49.6% for language only. On two-arm RoboDojo, VLM-predicted gripper poses score 0.7% to 3.3% while calling the VLA scores 40% to 45%. Their takeaway: VLMs know what to do but not how to move, so send sub-goals (best with a 2D target point) or let strong VLMs write code that queries 3D perception tools. The other six studies cover context, call timing, templates against free-form code, code evolution and run-time rescue. They matter for the agent side but not for the command format.

**Do they retrain per format?** No. No model is trained or fine-tuned in the study; all VLAs are frozen public checkpoints. The language-only versus target-marked comparison does use two different checkpoints, "MME-VLA, language-only sub-goal" and "MME-VLA, target-marked sub-goal", from the RoboMME authors' suite. So each format runs on a policy trained for it, but those policies came from another group, and the paper does not say whether they were trained on matched data and compute. They tested no 3D point or pose-path format sent to a learned policy. 3D targets go only to motion primitives.

**Relevance to us.** This is the strongest evidence that a 2D point beats plain language, but it covers only two formats, on one benchmark, using someone else's checkpoints. Our gap: five formats, each with a policy retrained under matched data and compute.

## GR00T 2 white paper (NVIDIA, November 2026)

NVIDIA. "GR00T 2: Scaling Robot Foundation Models for Generalization, Dexterity, and Adaptation." White paper PDF: https://research.nvidia.com/labs/gear/gr00t-2/assets/gr00t2_whitepaper.pdf (project page https://research.nvidia.com/labs/gear/gr00t-2). I did not find an arXiv ID.

**What was read.** Read in full: Section 2 (overview), Section 3.4 (Agentic GR00T 2), Appendix G (G.1 skill composition, G.2 visual steering, G.3). Skimmed: the table of contents and introduction. Not read: Sections 3.1 to 3.3, 3.5, 4 and the other appendices (data, RL, humanoid, infrastructure).

**What they test.** GR00T 2 is a 2B-parameter world-action model built on Cosmos-Predict2.5. It takes camera video, robot state and a language instruction, and predicts future frames plus an action chunk. Section 3.4 states the interface question directly: "what the interface between them should be, that is, what the agent sends to GR00T 2 for each step; we refer to this interface as the WAM API." They compare three settings on real YAM (two-gripper) and Dexter (two-hand) robots, with GPT-6 Astra as the agent:
1. GR00T 2 given the original task instruction.
2. GR00T 2 given the agent's per-step language instruction.
3. A point-conditioned GR00T 2 given the agent's per-step language plus a grasp point and a release point, drawn as arm-colored markers on one camera image.

Results (Table 22): YAM short-horizon pass@1 41.7% / 55.6% / 83.3%; Dexter 29.7% / 32.8% / 70.3%; YAM trash sorting 0/20 / 0/20 / 17/20; Dexter trash sorting 0/20 / 1/20 / 9/20. Points help most on spatial, exclusion and ordinal references. They help least on grasp and action references, because the markers say where but not how. Points also cut wrong-bin releases sharply. Agent cost is about $0.14 to $0.35 per episode, and the robot waits for the agent 29% to 38% of the time (Table 24). A simulation test on 45 LIBERO-PRO tasks uses an earlier, richer interface: a time-colored end-effector path plus a disc at the next grasp or release point, with markers computed from the simulator. It raises success from 69.6% (language-only checkpoint) to 85.0%. The same point-trained checkpoint with nothing drawn gets 72.5%. Separately, Section G.1 runs GR00T 2 as a language-prompted MCP tool inside code-as-policy programs. It reports 88.0% on RoboCasa365 with RL plus the agent, against 85.6% for HexaShell 1.5 on the same leaderboard (Table 19).

**Do they retrain per format?** Yes, for two formats. Each point-conditioned checkpoint is mid-trained from the pretrained model for 20k steps. "For each, we also train a language-only checkpoint on the same data, cameras, schedule, and compute, without markers." Point labels come automatically from grasp and release events in existing demos, projected into the camera. This is the same matched-retraining design we plan, but with only language against 2D points (plus the path variant in simulation). They did not test structured text, 3D points or a 3D pose path as separate retrained formats. The simulation path variant was compared only against language.

**Relevance to us.** This is the closest prior work to our design, and we must cite and position against it. Our difference: five formats instead of two, including 3D formats; all in one benchmark (RoboCasa); one small policy (DEM); and full success/time/token accounting against the agent alone.

## Shorter entries

- **RT-Trajectory** (arXiv 2311.01977). Abstract only. Conditions a policy on a rough 2D trajectory sketch drawn on the image. It generalizes to new tasks better than language- or goal-image-conditioned policies. The sketch can come from humans, image generators or waypoint generators. This is the 2D path format, trained into the policy.
- **HAMSTER** (arXiv 2502.05485). Abstract only. A hierarchical VLA in which a fine-tuned VLM outputs a coarse 2D end-effector path that guides a 3D-aware low-level policy. It reports about 20% average gain across seven generalization axes on real robots. Fixes one interface (2D path) rather than comparing formats.
- **MolmoAct** (arXiv 2508.07917). Abstract only. One model reasons through depth-aware perception tokens, then an editable 2D trajectory trace, then actions. The trace makes it steerable. This is a single-model, not agent-plus-policy, design.
- **PointVLA** (arXiv 2503.07511). Abstract only. Despite the name, this is about point *cloud* input injected into a frozen VLA action expert, not point-conditioned commands. Probably not relevant to the command format question.
- **RoboMME / MME-VLA.** Not read. X-Harness uses its language-only and target-marked sub-goal checkpoints, so its paper may describe how those two policies were trained. Worth finding.
- **GAE (pose-path conditioning).** Not found. `docs/OVERNIGHT.md` cites "GAE" for pose-path conditioning and noise on points and poses. A web search did not identify the paper, and I did not guess. Adi should supply the full title or arXiv ID.

## GAE (arXiv 2510.03896)
"GAE: Unleashing Physical Potential of VLM with Generalizable Action Expert", Liu et al., ICML 2026 (per Adi's notes). Read in full by Adi with Claude on Nov 26 (notes outside this repo); this entry restates only what those notes say. A fine-tuned VLM writes a short path of 3D gripper poses; a separate policy, trained once on 150k trajectories and then frozen, follows the path while seeing only an uncolored point cloud. Training adds noise to the poses. Its success drops from 0.57 to 0.33 at 10 cm of depth noise. No LLM agent, tools or time per task. It does not retrain the policy per command format. Our `pose_path` and `points_3d` formats are in its style; our noise settings (1 cm and 10 cm depth noise) follow it.
