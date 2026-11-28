#!/bin/bash
# The autonomous Claude Code session on Bouchet's agent partition (backlog item 9), so a sleeping Mac
# cannot stop the project. Starts tmux on the compute node, runs setup/agent_loop.sh inside it (which
# restarts Claude if it exits), and keeps the job alive while the tmux session lives.
# Read setup/agent_session_README.md first: Adi must log in to Claude once himself.
#
#   sbatch setup/agent_session.sbatch                                   (the real session, 7 days)
#   sbatch --time=00:15:00 --export=ALL,AGENT_MODE=check setup/agent_session.sbatch   (dry run: report only)
#
# Options (pass with --export=ALL,NAME=value):
#   AGENT_MODE=run|check   check = test modules, tmux, claude --version, network, Slurm, git; start nothing.
#   AGENT_WORKDIR=path     the git clone Claude works in (default $A/rua-harness-git).
#   AGENT_PUSH=1           push the clone's branch to GitHub every 10 minutes from this job, outside the
#                          YCRC sandbox, with Adi's GitHub key. Off by default; see the README.
#SBATCH --partition=agent
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=claude_agent
#SBATCH --output=/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/logs/agent_session_%j.out
set -u
P=/nfs/roberts/project/pi_dr877/ak3288
A=$P/apollo-lab
export AGENT_WORKDIR=${AGENT_WORKDIR:-$A/rua-harness-git}
export AGENT_LOGDIR=$A/logs
MODE=${AGENT_MODE:-run}
SESSION=claude
JOB=${SLURM_JOB_ID:-nojob}
say() { echo "$(date '+%a %b %d %H:%M:%S') $*"; }

say "mode=$MODE host=$(hostname -s) job=$JOB workdir=$AGENT_WORKDIR CLUSTER=${CLUSTER:-unset}"
module load claude >/dev/null 2>&1 || say "FAIL: module load claude"

check() {  # name, command...
  local name=$1; shift
  if out=$("$@" 2>&1); then say "ok    $name: $(echo "$out" | head -1)"; else say "FAIL  $name: $(echo "$out" | tail -1)"; fi
}

if [ "$MODE" = check ]; then
  SIF=${EBROOTCLAUDE:-/nonexistent}/claude-mod.sif
  SLURMB=/opt/slurm/current,/var/lib/slurmconf,/run/munge
  check "tmux on node"            tmux -V
  check "claude module"           bash -c 'echo $EBVERSIONCLAUDE; test -n "$EBVERSIONCLAUDE"'
  check "claude --version"        bash -c "cd '$AGENT_WORKDIR' && claude --version"
  check "claude logged in"        test -s "$HOME/.claude/.credentials.json"
  check "net (node) anthropic"    curl -sS -o /dev/null -w '%{http_code}' --max-time 15 https://api.anthropic.com/v1/messages
  check "net (node) github"       git ls-remote --heads https://github.com/Aditya100907/rua-harness.git overnight-nov26
  check "net (node) pypi"         curl -sS -o /dev/null -w '%{http_code}' --max-time 15 https://pypi.org/simple/
  check "net (container) anthropic" apptainer exec --contain "$SIF" curl -sS -o /dev/null -w '%{http_code}' --max-time 15 https://api.anthropic.com/v1/messages
  check "net (container) github"  apptainer exec --contain "$SIF" git ls-remote --heads https://github.com/Aditya100907/rua-harness.git overnight-nov26
  check "net (container) huggingface" apptainer exec --contain "$SIF" curl -sS -o /dev/null -w '%{http_code}' --max-time 15 https://huggingface.co
  check "slurm (node) squeue"     bash -c "squeue -u $USER -h | wc -l"
  check "slurm (node) sbatch --test-only" sbatch --test-only -p gpu_devel --gpus=1 -t 5 --wrap=true
  check "slurm (container) squeue" apptainer exec --contain --bind $SLURMB --env PATH=/opt/slurm/current/bin:/usr/bin:/bin "$SIF" squeue -u $USER -h -o %i
  check "slurm (container) sbatch --test-only" apptainer exec --contain --bind $SLURMB --env PATH=/opt/slurm/current/bin:/usr/bin:/bin "$SIF" sbatch --test-only -p gpu_devel --gpus=1 -t 5 --wrap=true
  check "git clone present"       git -C "$AGENT_WORKDIR" rev-parse --abbrev-ref HEAD
  check "git push --dry-run (node)" bash -c "cd '$AGENT_WORKDIR' && GIT_TERMINAL_PROMPT=0 GIT_SSH_COMMAND='ssh -o BatchMode=yes' git push --dry-run origin HEAD"
  # tmux: start a detached session that runs the module's claude --version, read its output, close it.
  T=$AGENT_LOGDIR/agent_check_tmux_$JOB.txt
  tmux new-session -d -s agentcheck "bash -lc 'module load claude; cd $AGENT_WORKDIR && claude --version > $T 2>&1'"
  sleep 60; tmux kill-session -t agentcheck 2>/dev/null
  check "tmux runs claude"        bash -c "cat $T 2>/dev/null | grep -v '^$' | tail -1; grep -qi 'claude' $T"
  say "check done"
  exit 0
fi

# --- run mode ---
if [ ! -s "$HOME/.claude/.credentials.json" ]; then
  say "Claude is not logged in on Bouchet (no ~/.claude/.credentials.json). Adi must log in once; see setup/agent_session_README.md. Exiting."
  exit 1
fi
if [ ! -d "$AGENT_WORKDIR/.git" ]; then say "no git clone at $AGENT_WORKDIR. Exiting."; exit 1; fi
rm -f "$AGENT_LOGDIR/agent_session.STOP"

tmux new-session -d -s $SESSION -x 220 -y 50 "bash $AGENT_WORKDIR/setup/agent_loop.sh"
tmux pipe-pane -t $SESSION -o "cat >> $AGENT_LOGDIR/agent_tmux_$JOB.log"
say "tmux session '$SESSION' started on $(hostname -s). Attach: ssh -t $(hostname -s) tmux attach -t $SESSION"

trap 'say "job ending: closing tmux"; tmux kill-session -t $SESSION 2>/dev/null; exit 0' TERM INT
i=0
while tmux has-session -t $SESSION 2>/dev/null; do
  sleep 60 & wait $!
  i=$((i+1))
  if [ "${AGENT_PUSH:-0}" = 1 ] && [ $((i % 10)) = 0 ]; then
    B=$(git -C "$AGENT_WORKDIR" rev-parse --abbrev-ref HEAD)
    if [ "$B" = main ] || [ "$B" = HEAD ]; then
      say "push skipped: on '$B' (never push main)"
    elif [ -n "$(git -C "$AGENT_WORKDIR" log --oneline "origin/$B..$B" 2>/dev/null)" ]; then
      ( cd "$AGENT_WORKDIR" && GIT_TERMINAL_PROMPT=0 GIT_SSH_COMMAND='ssh -o BatchMode=yes' git push -q origin "$B" ) \
        && say "pushed $B $(git -C "$AGENT_WORKDIR" rev-parse --short HEAD)" || say "push FAILED for $B"
    fi
  fi
done
say "tmux session ended; job exiting"
