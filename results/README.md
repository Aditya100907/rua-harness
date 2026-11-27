# results/

One CSV per run, named `YYYY-MM-DD_<condition>_<short-note>.csv`.

Columns: `task, episode, seed, success, seconds, llm_calls, input_tokens, output_tokens, cost_usd, code_commit`

Conditions: `policy_only`, `agent_only`, `agent_plus_policy`.

`scripted` (and `scripted_<testset>`): the hand-written controller in `policy/scripted.py`. It
reads simulator ground truth, so it is the "perfect command" oracle, not an agent result. See
`docs/testsets.md`.
