#!/bin/bash
# Submit queued sbatch commands (one per line in $1) whenever this user has < 3 jobs in SLURM.
# Lines are consumed from the top; log goes to logs/queue.log. Lightweight: only sleeps + sbatch.
Q=$1
while [ -s "$Q" ]; do
  n=$(squeue -u $USER -h | wc -l)
  if [ "$n" -lt 3 ]; then
    cmd=$(head -1 "$Q"); sed -i '1d' "$Q"
    out=$(eval "$cmd" 2>&1); echo "$(date +%H:%M) $out :: ${cmd:0:160}" >> logs/queue.log
  fi
  sleep 30
done
echo "$(date +%H:%M) queue empty" >> logs/queue.log
