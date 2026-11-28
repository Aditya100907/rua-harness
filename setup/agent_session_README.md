# The autonomous Claude session on Bouchet

Backlog item 9: run the autonomous Claude Code session on Bouchet's `agent` partition instead of Adi's Mac, so a sleeping Mac or an expired Duo login cannot stop the project. Tested on Sat Nov 28, 2026 (dry-run jobs 34965981 and 34966031, run-mode refusal job 34966229). Nothing here logged in to any account.

Files:
- `setup/agent_session.sbatch`: the job. 7 days, 1 CPU, 8 GB on `agent`. Starts tmux on the compute node, runs `setup/agent_loop.sh` inside it, writes the tmux screen to a log, optionally pushes the branch to GitHub every 10 minutes.
- `setup/agent_loop.sh`: runs inside tmux. Loads the YCRC `claude` module, starts Claude in the git clone with a prompt that points it to `docs/STANDING_ORDERS.md`, and restarts it when it exits (`--continue` keeps the conversation). Backs off 30 minutes after 5 exits in 15 minutes (usage limits).
- `scripts/on_bouchet.sh`: sourced by `scripts/status.sh`, `scripts/sync_bouchet.sh` and `scripts/ssh_duo_alert.sh` so they run directly on Bouchet instead of through `ssh bouchet`.

Logs, all in `/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/logs/`: `agent_session_<job>.out` (the job), `agent_loop.log` (each Claude start and exit), `agent_tmux_<job>.log` (the raw tmux screen, with terminal codes).

## What was tested and what works

| Check | Result |
|---|---|
| `agent` partition limits | 7 days max, 1 CPU and 8 GB per job (QoS `part_agent`), 2 running jobs per user. The watcher already uses 1 of the 2. |
| `module load claude` (version 2.1.314-0.1) and `claude --version` on an agent node | works, also inside tmux |
| tmux on the agent node | 3.2a, works. tmux is not inside the Claude container, so it runs in the job shell |
| Outbound network from the node and from inside the Claude container | works: api.anthropic.com, github.com (`git ls-remote`), pypi.org, huggingface.co |
| `squeue` and `sbatch --test-only` from the node and from inside the Claude container | works (with the Slurm binds the YCRC wrapper uses) |
| ssh from the login node to the agent node where a job of yours runs | works (needed to attach to tmux) |
| `scripts/status.sh` on the login node | works, no ssh |
| `scripts/sync_bouchet.sh` from the git clone on Bouchet | works as a local rsync (tested with `SYNC_DRYRUN=1`, nothing copied) |
| `scripts/ssh_duo_alert.sh` on Bouchet | exits at once: no SSH connection to watch, no Duo needed |
| Run mode without a Claude login | refuses to start and says why |
| Claude logged in on Bouchet | **no**: needs Adi |
| `git push` from Bouchet | **no**: no GitHub credential on Bouchet. Needs Adi (below) |

The git clone is at `/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness-git` (branch `overnight-nov26`, cloned over HTTPS since the repo is public, `user.name`/`user.email` set to the Mac's). The rsync copy `apollo-lab/rua-harness` that jobs run from is unchanged.

## What the YCRC sandbox allows (read from the module's own files)

The module runs Claude Code in an Apptainer container (`/apps/software/system/software/claude/2.1.314-0.1/`). It sees only the folder it starts in, the folders passed with `--bind`, `~/.claude`, conda, `/apps` and Slurm. Your home folder, `~/.ssh` and environment variables with names like `*TOKEN*`, `*API*`, `*_KEY` are hidden. It refuses to start on a login node.

Its managed settings (cannot be changed by us):
- Permission mode is `auto`; bypass mode is disabled.
- Always allowed: `squeue`, `sacct`, `sinfo`, `module`, `conda activate`.
- **Asks a person first:** `rsync`, `ssh`, `scp`, `kill`, `pkill`, `tmux`, `git push`, `conda remove`.
- **Denied:** `scancel`, `scontrol`, `apptainer`, `crontab`, reading `.env`, `.ssh`, `*credentials*`, `*.key`.

What this means for the autonomous session:
- It can submit jobs (`sbatch`) and watch them (`squeue`, `sacct`) directly.
- It cannot cancel jobs (`scancel` is denied). It must leave a cancel request in `docs/decisions_needed.md`, or let jobs hit their time limit.
- An "ask" command stops the session until someone answers in tmux. The session should avoid them: commit locally and let the job push (below), and prefer running code from the git clone over rsync (see "Making the clone the jobs' folder").
- Do not try to get around these rules (for example by hiding `scancel` in a script). They are YCRC's policy for this module.

## Commands Adi must run once himself

Run them in a terminal on the Mac. Each needs your login, so the session did not run them.

### 1. Log in to Claude on Bouchet (needs your Anthropic account)
Use a Claude subscription login (Pro or Max), not an API key: an API key costs money per call, which the standing orders forbid without your say-so. A 24/7 session will hit the plan's usage limits; the loop waits and retries.
```
ssh bouchet
salloc -p agent --mem=4G -t 0:30:00          # uses the 2nd agent slot for 30 minutes
module load claude
cd /nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness-git
claude                                       # follow the login prompt; open the URL on the Mac
# when Claude shows its prompt: type /exit, then
exit                                         # give the agent slot back
```
The login is saved in `~/.claude/` on Bouchet and reused by every later session.

### 2. Let the job push to GitHub (choose one)
Inside the sandbox Claude cannot read `~/.ssh` and `git push` needs a person's approval. So the job itself (outside the sandbox) pushes the branch every 10 minutes when started with `AGENT_PUSH=1`. It never pushes `main`. It needs a GitHub credential that only you can create.

**Option A (recommended): a deploy key for this one repo.**
```
ssh bouchet
ssh-keygen -t ed25519 -f ~/.ssh/rua_harness_deploy -N "" -C "bouchet rua-harness deploy"
cat ~/.ssh/rua_harness_deploy.pub
```
On GitHub: repo Settings, Deploy keys, Add deploy key, paste it, tick "Allow write access". Then:
```
cat >> ~/.ssh/config <<'EOF'
Host github.com
  IdentityFile ~/.ssh/rua_harness_deploy
  IdentitiesOnly yes
EOF
git -C /nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness-git remote set-url origin git@github.com:Aditya100907/rua-harness.git
git -C /nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness-git push --dry-run origin overnight-nov26
```
A deploy key can push any branch, including `main`, and `main` has no branch protection now. Turn it on (Settings, Branches, add a rule for `main`, require a pull request) so only you can merge.

**Option B: no push from Bouchet.** Leave `AGENT_PUSH` off. Commits stay in the clone on Bouchet; pull them from the Mac with `git pull bouchet-clone overnight-nov26` after `git remote add bouchet-clone bouchet:/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness-git` (works only while your Mac's ssh connection is approved).

Pushing from the job, outside the sandbox, means the YCRC "ask before git push" prompt never appears for these pushes. That is your choice to make; option B avoids it.

### 3. Stop the Mac session, then start the cluster session
Only one autonomous session may run at a time, or both will sync code and launch jobs. Stop the Mac session first (its timers and `caffeinate`), then:
```
ssh bouchet
cd /nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness-git && git pull
squeue -u ak3288 -p agent                    # at most 1 job (the watcher), or there is no free slot
sbatch --export=ALL,AGENT_PUSH=1 setup/agent_session.sbatch      # drop AGENT_PUSH=1 for option B
```
To repeat the dry run (starts nothing, prints a ok/FAIL list in the job log):
```
sbatch --time=00:10:00 --export=ALL,AGENT_MODE=check setup/agent_session.sbatch
```

## Watching, steering, stopping

Watch or talk to it (attach to its tmux):
```
ssh bouchet
NODE=$(squeue -u ak3288 -n claude_agent -h -o %N); ssh -t $NODE tmux attach -t claude
```
Detach without stopping it: `Ctrl+b` then `d`. Answer any "ask" prompt while attached. Without attaching: `tail -f /nfs/roberts/project/pi_dr877/ak3288/apollo-lab/logs/agent_loop.log`.

Stop it for good:
```
ssh bouchet
touch /nfs/roberts/project/pi_dr877/ak3288/apollo-lab/logs/agent_session.STOP   # the loop will not restart Claude
scancel -n claude_agent -u ak3288
```
The job lasts 7 days at most. Resubmit the `sbatch` line to keep going; `--continue` picks up the last conversation.

## Making the clone the jobs' folder (optional, later)

Today jobs run from the rsync copy `apollo-lab/rua-harness`, and the session copies code there with `bash scripts/sync_bouchet.sh` (a local rsync). Once the session runs on Bouchet, the copy step can go away by making the clone the jobs' folder. Do this only when no job is starting, since paths change under it:
```
cd /nfs/roberts/project/pi_dr877/ak3288/apollo-lab
rsync -a --ignore-existing rua-harness/results/ rua-harness-git/results/    # keep results written on the cluster
mv rua-harness rua-harness-rsync-20261128 && mv rua-harness-git rua-harness
```
Check first what else jobs write into the rsync copy (`git -C rua-harness-git status` after copying it over shows untracked files). The launcher then uses `apollo-lab/rua-harness` automatically, `sync_bouchet.sh` becomes a no-op, and runs record the commit from git. Nothing is deleted.

## Running on Bouchet (for the session itself)

- You are on the cluster. Run `squeue`, `sacct`, `sbatch` directly. Never `ssh bouchet`.
- No `caffeinate`, no Duo alert, no Mac paths (`~/Desktop/APOLLO-Lab/` is not here, so the daily `PROJECT_CONTEXT.md` entry must wait for the Mac).
- Edit and commit in the clone. Run `bash scripts/sync_bouchet.sh` before submitting jobs that need new code.
- `scancel` is denied: write cancel requests in `docs/decisions_needed.md`.
- Each Bash command starts a fresh shell: chain `module load miniconda; conda activate /nfs/roberts/project/pi_dr877/ak3288/envs/dem; ...` in one command.
- Checkpoints and the conda env are read-only to you; scratch and `apollo-lab/` are writable.
- Keep the 10-minute timer: the standing orders still apply.
