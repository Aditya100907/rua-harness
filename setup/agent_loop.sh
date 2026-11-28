#!/bin/bash
# Runs inside the tmux session that setup/agent_session.sbatch starts. Starts Claude Code (YCRC claude
# module) in the git clone and restarts it when it exits. To stop for good: touch the STOP file below,
# then /exit in Claude (or scancel the job).
P=/nfs/roberts/project/pi_dr877/ak3288
A=$P/apollo-lab
S=/nfs/roberts/scratch/pi_dr877/ak3288
WORK=${AGENT_WORKDIR:-$A/rua-harness-git}
LOG=${AGENT_LOGDIR:-$A/logs}
STOP=$LOG/agent_session.STOP
module load claude
cd "$WORK" || exit 1

# Folders Claude may use besides the clone. ro unless :rw. Checkpoints stay read-only: training and
# evaluation run as separate sbatch jobs outside the sandbox, which write them.
BINDS=(--bind="$A:rw" --bind="$P/checkpoints:ro" --bind="$P/envs:ro" --bind="$S:rw")

PROMPT="You are the autonomous session for this project, now running on Bouchet's agent partition (job ${SLURM_JOB_ID:-?}), not on Adi's Mac. Read docs/STANDING_ORDERS.md, CLAUDE.md and the 'Running on Bouchet' section of setup/agent_session_README.md first, then continue the backlog. You are already on the cluster: call squeue and sbatch directly, never ssh bouchet; caffeinate and the Duo alert are not needed. Edit and commit in this git clone ($WORK); run bash scripts/sync_bouchet.sh to copy code to the folder jobs run from."

n=0; recent=()
while true; do
  [ -e "$STOP" ] && { echo "$(date) STOP file present; not restarting" | tee -a "$LOG/agent_loop.log"; break; }
  n=$((n+1))
  echo "$(date) start $n" >> "$LOG/agent_loop.log"
  if [ $n = 1 ] && [ "${AGENT_FRESH:-0}" = 1 ]; then
    claude "${BINDS[@]}" "$PROMPT"
  else
    # Continue the last conversation in this folder; fall back to a fresh one if there is none.
    claude "${BINDS[@]}" --continue "$PROMPT" || claude "${BINDS[@]}" "$PROMPT"
  fi
  rc=$?
  echo "$(date) claude exited rc=$rc" >> "$LOG/agent_loop.log"
  # Back off if it keeps dying (for example a usage limit): 5 exits in 15 minutes -> wait 30 minutes.
  now=$(date +%s); recent=($(for t in "${recent[@]}" $now; do [ $((now - t)) -lt 900 ] && echo $t; done))
  if [ ${#recent[@]} -ge 5 ]; then echo "$(date) 5 exits in 15 min; sleeping 30 min" >> "$LOG/agent_loop.log"; sleep 1800; recent=(); else sleep 60; fi
done
