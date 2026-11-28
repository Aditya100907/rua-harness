#!/bin/bash
# One screen: running jobs, latest checkpoint and loss per run, latest success numbers, watcher.
#   bash scripts/status.sh            (from the Mac through ssh bouchet, or directly on Bouchet)
source "$(dirname "$0")/on_bouchet.sh"
if [ "$ON_BOUCHET" = 1 ]; then RUN=(bash -s); else RUN=(ssh bouchet 'bash -s'); fi
"${RUN[@]}" <<'REMOTE'
P=/nfs/roberts/project/pi_dr877/ak3288; R=$P/apollo-lab
echo "== jobs (GPU and agent)"; squeue -u ak3288 -h -o "%.10i %.24j %.2t %.10M %.12P %R" | grep -v " day " ; echo "   CPU jobs on day: $(squeue -u ak3288 -h -p day | wc -l)"
echo "== runs: last checkpoint, step, loss, steps/s"
for d in $P/checkpoints/*/; do
  r=$(basename $d); ck=$(ls $d/ckpt_*.pt 2>/dev/null | tail -1); l=$(tail -1 $d/train_log.csv 2>/dev/null | cut -d, -f1,2,5)
  printf "  %-28s %-22s %s\n" $r "$(basename ${ck:-none})" "$l"
done
echo "== latest results (success / episodes)"
for f in $(ls -t $R/rua-harness/results/*.csv 2>/dev/null | head -12); do
  python3 -c "import csv,sys;r=list(csv.DictReader(open('$f')));print('  %-70s %d/%d'%('$(basename $f)'[:70],sum(int(x['success']) for x in r),len(r)))"
done
echo "== watcher"; tail -3 $R/logs/watcher.log
echo "== queue"; grep -v '^#' $R/runs/queue.txt | cut -d'|' -f1
REMOTE
