#!/bin/bash
# usage: jobs/bw_retry.sh <job name> <ARGS for ce.py> [log]   (ER_WORK, PY from the environment;
#        SB = sbatch script, default jobs/gpubw.sbatch; MERGE passed through for jobs/gpu2bw.sbatch)
# submits jobs/gpubw.sbatch on cn07; if it landed on a non-Blackwell GPU (exit 3), resubmits after 2 min (<= 12 tries).
N=$1; A=$2; L=${3:-logs/bw_retry_$N.log}
cd /csehome/m25csa019/amazon_ml/er
for i in $(seq 1 12); do
  until [ "$(squeue -u $USER -h | wc -l)" -lt 3 ]; do sleep 30; done
  j=$(sbatch --parsable -J "$N" -w cn07 --export=ALL,ER_WORK=$ER_WORK,PY=$PY,ARGS="$A",MERGE="$MERGE" ${SB:-jobs/gpubw.sbatch}) || { sleep 60; continue; }
  echo "$(date +%H:%M) try $i submitted $j" >> $L
  until [ -n "$(sacct -j $j -X -n --format=State | grep -E 'COMPLETED|FAIL|CANCEL|TIMEOUT|OUT_OF')" ]; do sleep 30; done
  st=$(sacct -j $j -X -n --format=State,ExitCode | xargs)
  echo "$(date +%H:%M) $j $st" >> $L
  case "$st" in COMPLETED*) exit 0;; *" 3:0"*) sleep 120;; *) exit 1;; esac
done
echo "$(date +%H:%M) no Blackwell after 12 tries" >> $L; exit 2
