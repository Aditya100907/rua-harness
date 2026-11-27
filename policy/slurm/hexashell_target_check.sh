#!/bin/bash
# HexaShell target split vs our evaluation's scenes (policy/hexashell_target_check.py), on a CPU node:
#   sbatch --export=ALL,CMD="bash $REPO/policy/slurm/hexashell_target_check.sh" policy/slurm/cpu_job.sbatch
# Needs setup/setup_hexashell_bouchet.sh hexashell first. Seeds 1000-1049 for the 3 smoke tasks
# (the report uses 1000-1004; the rest show how often a seed's scene changes between runs).
set -uo pipefail
HX=$PROJ/apollo-lab/HexaShell
HPY=$PROJ/envs/hexashell-robocasa/bin/python
OUT=$ROOT/runs/hexashell_target_check
mkdir -p "$OUT"
SEEDS=$(seq 1000 1049)
C="python -m policy.hexashell_target_check"
$C dump --side ours --seeds $SEEDS --out "$OUT/ours_a.json"
$C dump --side ours --seeds $SEEDS --out "$OUT/ours_b.json"
$C dump --side ours --counter-fix --seeds $SEEDS --out "$OUT/ours_fix.json"
( set -a; source "$PROJ/envs/hexashell-robocasa/bench-env.sh"; set +a
  "$HPY" -m policy.hexashell_target_check dump --side hexa --hexashell "$HX" --seeds $SEEDS --out "$OUT/hexa_a.json"
  "$HPY" -m policy.hexashell_target_check dump --side hexa --hexashell "$HX" --seeds $SEEDS --out "$OUT/hexa_b.json" )
for pair in "ours_a hexa_a" "ours_a ours_b" "hexa_a hexa_b" "ours_fix hexa_a" "ours_b hexa_a"; do
    set -- $pair
    echo "=== $1 vs $2"
    $C compare "$OUT/$1.json" "$OUT/$2.json" --csv "$OUT/$1_vs_$2.csv" | tail -1
done
