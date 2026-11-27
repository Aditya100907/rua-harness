# Decisions only Adi can make

Each with the options, the evidence so far, and a recommendation. Updated through the night.

## 1. HexaShell already runs on RoboCasa. Does William still port it? (new, Nov 27 00:30)
**Finding.** The public HexaShell repo (https://github.com/HexaFuture/HexaShell, commit `914d482`, Nov 25) now contains a full RoboCasa port (`hexashell/benches/robocasa/`: sim, host, control, protocol), written per-task skills for all 50 RoboCasa365 tasks (`skills/robocasa/*.md`), and results: 59.3% without skills and 85.6% with skills on RoboCasa365 (1,500 episodes, pretrain50 split, test seeds 20 to 49), #1 on the leaderboard as of Nov 1. Its RoboCasa is 1.0.1 at `b4684e6` with patches (ours: `425684d`, unpatched); robosuite `5ce6643` matches ours; MuJoCo 3.3.1 and numpy 2.2.5 match; it runs CPython 3.12 and cameras at 512 x 512. `docs/agent_port_spec.md` assumes the repo "has the 14 tools but not the harness"; that is no longer true. I read the README and listed the files; I did not run it.
**Options.**
- A. William uses HexaShell's own RoboCasa port as the agent-alone baseline and adds `run_skill` as a 15th call. Saves most of his port work; the baseline is the published system.
- B. William continues his own port (as specified). Costs days; differences from the published port need explaining.
- C. Use HexaShell's published numbers only. Not comparable (different split, seeds, skills).
**Recommendation: A.** Run HexaShell's port unchanged on our split and seeds (target split, atomic-seen 18 tasks), with and without skills, and add `run_skill` through its tool mechanism. Check its RoboCasa patches do not change success checks. William should read `hexashell/benches/robocasa/` first.
**Also affects the claim:** an agent with general tools already reaches 59 to 86% on RoboCasa365. The policy's value has to show as time and tokens at similar success, or as success on tasks where the agent's motion tools fail.

## 2. run_skill stops on the simulator's success check
The policy has no learned "done" output. `policy/run_skill.py` stops when `env._check_success()` turns true and hides that flag from the agent (`grading_only`). This leaks a little ground truth (the skill ends early when it succeeds). Options: keep it for now and train a done head (backlog), or run each skill to its time limit. **Recommendation:** keep it for the weekend run, report it, train a done head next week.

## 3. Cameras for the policy
The plan says left and right agent views; the DEM README example uses a scene camera plus the wrist. Run 5 (left + wrist) answers this on the smoke tasks by about 05:00. **Recommendation:** decide from run 1 vs run 5 at 20k steps.

## 4. Which claim the paper makes
Not chosen. Evidence will come from F4/F6 (formats with perfect commands) and Part V (agent-written commands). Keep all options open until those exist.

## 5. Which planner LLM runs the RPent (Harness VLA) baseline (new, Nov 27 01:00)
**Finding.** RPent is installed on Bouchet (`envs/rpent`), the RLDX-1 VLA weights are downloaded (public, not gated, RLWRLD non-commercial license), and its RoboCasa simulator checks pass on a gpu_devel GPU. SAM 3 is gated but RoboCasa does not use it. The only missing piece for a logged episode is a planner LLM, and every option either costs money or needs a terms check. Details: `docs/rpent_notes.md`.
**Options.**
- A. Codex with Adi's ChatGPT plan login (`codex login --device-auth` on Bouchet, `CODEX_HOME` in project space). Matches RPent's reference planner (Codex, GPT-5.5, xhigh). Adi must check that the plan terms and rate limits allow unattended harness runs (180 atomic runs, up to 100 tool calls each).
- B. A paid API key (OpenAI, Anthropic or Gemini) in `.env` on the cluster. Clear terms, costs per token. Needs a budget: start with 1 episode, then the 3 smoke tasks.
- C. Claude Code with Adi's claude.ai login. The Agent SDK docs say claude.ai login is not allowed for agents built on the SDK without approval. Not recommended.
- D. A local open model behind vLLM (free). Needs a second GPU and environment, and the result does not match published numbers.
**Recommendation: A if the plan terms allow it, else B with a small budget.** Run the no-LLM VLA check first (fix two stale lines in RPent's test, one gpu_devel job), then one OpenDrawer seed 1 episode.
