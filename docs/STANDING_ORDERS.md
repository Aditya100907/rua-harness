# Standing orders (from Adi, Sat Nov 28, 2026)

Every session reads this first. Written verbatim from Adi's message.

You now run this project with full autonomy. Write this message into
docs/STANDING_ORDERS.md and add a line to CLAUDE.md telling every session to
read it first. Then follow it.

AUTHORITY
You decide, without asking me: which experiments to run, their order, training
settings, cameras, loss, seeds, queue contents, code structure, bug fixes,
when a result is trusted, and what to try next. Record each decision and its
reason in docs/decisions_log.md. Do not wait for me on anything in this
paragraph, ever.

ONLY THESE NEED ME
Spending money. Sending a message to any person. Merging to main. Deleting
data or checkpoints. The paper's final claim (recommend one, with evidence).
For these, write the request in docs/decisions_needed.md and keep working on
something else.

NEVER STOP
- Never end a turn. Keep a 10-minute background timer running at all times and
  act when it fires: collect results, update findings, launch the next runs.
- Never wait on a helper or a job with nothing else in hand.
- GPUs must not sit idle: if fewer than 8 are busy and the queue is empty, add
  the next experiment from the backlog below.
- If Bouchet is unreachable, retry every 10 minutes and work from the offline
  list. At 7 hours into an SSH connection, alert me with a macOS notification
  and a spoken message that Duo is needed.
- Keep caffeinate running with no time limit. Log every stall and its cause in
  docs/autonomous_log.md.

BACKLOG, in priority order. Reorder as results demand.
1. Make points matter: name dropout, held-out objects, the distractor test on
   each new checkpoint. This is the main scientific question right now.
2. Format study at full scale: every format on all 18 tasks, 3 seeds, 300k.
3. How accurate must a command be: point-error sweeps at 0, 1, 2, 5, 10 cm,
   and training with point noise.
4. Agent writes the commands. Serve an open vision-language model with vLLM on
   Bouchet (free) and use it as the commander for every format. Measure
   success and pointing error against perfect commands.
5. Agent alone against agent plus policy: run HexaShell's RoboCasa port with
   the target split and the same open model, then add run_skill. Same seeds
   for both. If William's branch has no HexaShell results by Sunday noon, do
   this yourself and note it in docs/for_william.md.
6. Second benchmark: LIBERO, sentence format and the best point format.
7. Composite multi-step tasks, where an agent should help most.
8. Stock DEM weights: when the files appear, score them and test fine-tuning
   each format from them. MeanFlow debugging is low priority until then.
9. Move yourself to Bouchet's agent partition so a sleeping Mac cannot stop
   you. Do everything you can, test it, and leave me only the commands that
   need my login.

OFFLINE LIST, for when the cluster is unreachable
Analysis and plots from results already synced. Paper tables, figures and
draft sections. Code review, tests, refactors. Reading and related-work notes.
Preparing the next experiments so they launch the moment SSH returns.

AFTER EVERY RESULT
Update docs/findings.md, docs/research_journal.md, results/tables.md and the
paper's tables. Then choose the next experiment by information per GPU-hour.

RULES THAT STAY
One change per experiment. Three seeds before a claim. Every number with its
episode count and interval. Locked test set untouched until final runs. Check
any surprise over 15 points before believing it. At most 8 GPUs, none idle,
scoring on CPU nodes. No paid API calls. No secrets in the repo. Commit and
push after every item, on a branch, never to main.

REPORTS
Rewrite docs/DAILY.md at 9am and 9pm: what was learned, what is running, what
is next, what needs me. Append a dated entry to the log in
~/Desktop/APOLLO-Lab/PROJECT_CONTEXT.md once a day. Reread that file before
editing it and change only your own lines, since other sessions edit it too.

## Current priorities (Adi, Sat Nov 28, about 15:00). These are decisions. They replace the two
## additions that were here (the research-synthesis priority set and reading PROPOSALS.md each cycle).
1. System benchmark. William runs HexaShell alone and is on schedule; do not run HexaShell alone yourself.
   Your part: (b) the policy alone on the same seeds, and (c) build and test run_skill inside HexaShell's
   RoboCasa port so it is ready the moment his baseline lands. When his branch appears on GitHub, read it
   and line up seeds, model and split with his. Then run all three conditions on the 3 smoke tasks, then
   all 18: success, seconds, LLM calls, tokens.
2. Add Diffusion Policy as the end-to-end baseline: same demos, same evaluation.
3. The system table and the 18-task rows stay in the paper. They are the foundation.
4. The real robot stays in scope. Keep the 3D point-map track alive for it whatever its simulation score:
   clean reruns, depth-noise training, and a sim-to-real plan for the xArm 7 in docs/sim_to_real.md stating
   what must be true by Dec 15 to run it.
5. The interaction-layer experiments (twins, name dropout, tolerance sweeps, agent-written commands)
   continue on their GPU share.
GPU shares: interaction layer 4 (pool A), long runs (18-task rows, Diffusion Policy, seeds) 3 (pool B),
open model / HexaShell run_skill 1 (pool C), borrowing when a pool's queue is empty. The research folder
(~/Desktop/APOLLO-Lab/research/litreview-2026-11-27/, read-only) is an optional input for the interaction share.
