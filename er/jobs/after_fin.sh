#!/bin/bash
# When the final pipeline's last job completes, release it (copy to submissions/fin + version snapshot). Light: file copies only.
# usage: bash jobs/after_fin.sh [dag log, default logs/dag.log]
cd /csehome/m25csa019/amazon_ml/er
L=${1:-logs/dag.log}
until grep -q "^.* fin (.*) \(COMPLETED\|FAILED\|CANCELLED\|TIMEOUT\)" $L || grep -q "SKIPPED\|dag finished" $L; do sleep 120; done
if grep -q " fin (.*) COMPLETED" $L && grep -q "PASS" logs/fz_fin_*.out; then
  ER_WORK=/scratch/m25csa019/er_final /scratch/m25csa019/envs/er/bin/python src/snap.py tag=fin desc="final consolidated pipeline: France-fixed normalization + K100/expansion/ANN + 3 cross-encoders (cev, cem, ceb) + cascade stage 2" >> logs/after_fin.log 2>&1
  echo "$(date +%H:%M) fin released" >> logs/after_fin.log
else
  echo "$(date +%H:%M) fin not released (job failed or validator did not pass)" >> logs/after_fin.log
fi
