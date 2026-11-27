# Running a coding agent on Bouchet

Task M4. Can a coding agent (for example Claude Code) run on Bouchet itself, with internet access, so it can drive jobs without Adi's Mac staying on? Written 2026-11-26 from the YCRC docs only. Nothing was logged in to or tested on the cluster.

**Short answer: yes, and YCRC supports it.** Bouchet has an `agent` partition built for long coding-agent sessions. YCRC provides a sandboxed `claude` module (beta) that runs Claude Code with your own Anthropic account. You do not need to install Node or Claude Code yourself.

Labels: **[docs]** means stated in the YCRC docs at the cited URL. **[unverified]** means my inference, not confirmed by the docs.

## What the docs say

### The `agent` partition
Source: https://docs.ycrc.yale.edu/clusters/bouchet/ (partition list) and https://docs.ycrc.yale.edu/ai/aicodingtools/

- **[docs]** "Use the agent partition for interactive coding-agent workflows that need to remain active substantially longer than a typical interactive session."
- **[docs]** Limits:

  | Limit | Value |
  |---|---|
  | Max job time | 7 days (`7-00:00:00`) |
  | Default time | 1 hour |
  | Max running jobs per user | 2 |
  | Max CPUs per job | 1 |
  | Max memory per job | 8 GiB |
  | Default memory | 1 GiB per CPU |
  | Nodes | 2 nodes, Cascade Lake, 48 CPUs and 368 GiB each |

- **[docs]** The defaults are small, so ask for what you need: `salloc -p agent --mem=8G -t 8:00:00`.
- **[docs]** "Run coding agents on a compute node, never on a login node. YCRC's coding-agent modules refuse to start on a login node."
- **[docs]** An agent session is for editing, inspecting files and small tests. Larger work (environment builds, GPU work) should go out as separate Slurm jobs on the right partition. The agent can submit those or write the batch script. Those jobs run under your account and allocations.

### The YCRC `claude` module (commercial, beta)
Source: https://docs.ycrc.yale.edu/ai/commercial-coding-agents/

- **[docs]** "YCRC is currently beta-testing a Claude-only sandbox module that adds cluster-specific protections around Claude Code."
- **[docs]** Inference still happens at Anthropic. The module limits what Claude can see and do on the cluster. It runs Claude in an Apptainer container that shows only the working directory and the folders needed for supported tools. Managed settings apply YCRC command policies (some commands run automatically, some need approval, some are denied). SSH credentials are hidden, and the launcher removes environment variables that may hold tokens.
- **[docs]** Supported tools inside the sandbox: modules, Conda, R and Slurm. To add another tool, ask YCRC.
- **[docs]** Usage, from inside an agent-partition session:
  ```
  ml claude
  cd ~/scratch/some-work-folder   # any non-hidden subfolder of home, project, scratch or PI storage
  claude
  ```
  The module refuses to start from a location that fails its checks.
- **[docs]** "You need a Claude account with Anthropic to use the current beta module." If you have not used Claude on the cluster before, it asks you to log in. Later sessions reuse the saved login, sessions and settings.
- **[docs]** Data rules: until Claude Enterprise is available, commercial coding agents may be used only with **low-risk data**. Personal Claude accounts are not covered by the planned medium-risk approval.
- **[unverified]** Whether the `claude` sandbox allows general outbound internet (pip, git clone, HuggingFace, RoboCasa downloads). It must reach Anthropic to work at all, but the commercial page does not say whether other hosts are allowed. If they are blocked, do downloads in a normal shell or an `sbatch` job instead.
- **[unverified]** Whether `claude -p` (non-interactive mode) works with the `claude` module inside an `sbatch` job. The docs show batch mode only for the local module below.

### The YCRC `local-coding-agents` module (free, local model)
Source: https://docs.ycrc.yale.edu/ai/local-coding-agents/

- **[docs]** `module load local-coding-agents` gives four interfaces: `pi`, `copilot`, `codex` and `claude`. All four use a YCRC-hosted model (currently Qwen3.8-Flash-Next) instead of Anthropic. No account or API key is needed, and it costs nothing.
- **[docs]** The supported workflows explicitly include "Internet access, for example to install packages or clone repositories", plus Slurm, modules, Conda and git.
- **[docs]** Each command the agent runs starts in a fresh shell. `module load`, `conda activate`, `export` and `cd` do not carry over, so chain them in one command.
- **[docs]** `claude --bind /path1,/path2` gives the agent read-only access to other folders. Group shared folders (`/nfs/roberts/project/<group>/shared`) are visible automatically.
- **[docs]** Batch mode: `claude -p "task" --output-format json` from an `sbatch` script with `--partition=agent`. Batch jobs count toward the 2-session limit. In batch mode the agent cannot ask for approval, so actions that need approval are refused.
- **[unverified]** That the local model is strong enough for this project's work. It is a smaller model than Claude.

### Keeping a session alive
Source: https://docs.ycrc.yale.edu/clusters-at-yale/guides/tmux/ and https://docs.ycrc.yale.edu/clusters-at-yale/job-scheduling/

- **[docs]** If you disconnect, an `salloc` session is killed. Run `tmux` on the **login node** first, then `salloc` inside tmux, never the reverse. Detach with `Ctrl+b` then `d`; reattach with `tmux attach`.
- **[docs]** Bouchet has two login nodes. A tmux session lives on one of them, so if you cannot find it, check `hostname -s` and `ssh` to the other login node.

## One-time setup for Adi (netid ak3288)

Adi does these himself. Logging in to Anthropic needs his account. Nothing here was run.

1. **Check the data is low-risk.** RoboCasa simulator data and this repo's code look low-risk to me **[unverified: my judgment, not a YCRC ruling]**. Keep `.env` keys and anything from `~/Desktop/APOLLO-Lab/private/` off the cluster folder the agent works in.
2. **Decide how Claude is billed.** If the Claude account is a subscription (Pro or Max), sessions use that plan. If it is an API key, every call costs money, and the repo rule "no LLM API spending without Adi's say-so" applies. **[unverified]** which login options the YCRC module accepts.
3. **Connect and start tmux on a login node:**
   ```
   ssh bouchet
   hostname -s            # note which login node, for reattaching later
   tmux new -s agent
   ```
4. **Get an agent-partition session inside tmux:**
   ```
   salloc -p agent --mem=8G -t 7-00:00:00
   ```
   Ask for less than 7 days if you do not need it. This uses 1 of the 2 allowed agent sessions.
5. **Start Claude from the project folder** (a non-hidden subfolder of project storage, which the docs allow):
   ```
   ml claude
   cd /nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness
   claude
   ```
   On first start, follow the login prompt (it gives a URL to open on the Mac). Reattach later with `tmux attach -t agent`.
6. **Give it the scratch datasets if needed.** The docs show `--bind` for the local module. **[unverified]** whether the `claude` module accepts `--bind` too. Scratch is an allowed start location, but the agent sees only the folder it starts in plus approved folders.
7. **Test before relying on it** (each item is **[unverified]** until it passes):
   - `squeue -u ak3288` and a small `sbatch --partition=gpu_devel` job submitted from inside Claude.
   - `module load miniconda; conda activate /nfs/roberts/project/pi_dr877/ak3288/envs/dem; python -c "import robocasa"` in one command.
   - Internet: `curl -sI https://pypi.org` and `git ls-remote https://github.com/robocasa/robocasa`.
   - `git` push access. SSH keys are hidden from the sandbox, so pushes probably fail **[unverified]**. That fits the repo rule of editing on the Mac and rsyncing to the cluster.

### Not recommended: installing Claude Code yourself
You could install Node and Claude Code under project space (for example `npm install --prefix /nfs/roberts/project/pi_dr877/ak3288/tools/claude @anthropic-ai/claude-code`, which keeps the many small npm files out of the 500,000-file home quota). The YCRC docs do not describe this route. It would run outside the YCRC sandbox, and the docs say to check the tool's own security settings in that case. Prefer `ml claude`. If you do self-install, put the npm cache in project space too (`npm config set cache /nfs/roberts/project/pi_dr877/ak3288/.npm`). Both commands are **[unverified]** on Bouchet.

## Open questions to ask YCRC (support reopens Mon Nov 30)
- Does the `claude` module allow outbound internet beyond Anthropic?
- Does `claude -p` work with the `claude` module in an `sbatch` job on `agent`?
- Does the `claude` module accept `--bind` for scratch datasets?
- Which Anthropic login types does the beta accept (subscription, API key, or both)?
