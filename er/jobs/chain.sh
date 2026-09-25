#!/bin/bash
# Run sbatch commands (one per line in $1) strictly one after another: submit the next only after the previous
# job COMPLETED; stop on FAILED/CANCELLED/TIMEOUT. Waits while the per-user submit limit is reached.
# Lightweight: sleeps + sbatch/sacct only. Log: logs/chain.log
F=$1
while [ -s "$F" ]; do
  cmd=$(head -1 "$F")
  while [ "$(squeue -u $USER -h | wc -l)" -ge 3 ]; do sleep 30; done
  j=$(eval "$cmd" 2>&1); sed -i '1d' "$F"
  echo "$(date +%H:%M) submitted $j :: ${cmd:0:120}" >> logs/chain.log
  case "$j" in (*[!0-9]*|'') echo "$(date +%H:%M) submit failed, stopping" >> logs/chain.log; exit 1;; esac
  while true; do
    st=$(sacct -j "$j" -n -X --format=State | tr -d ' ')
    case "$st" in
      COMPLETED) echo "$(date +%H:%M) $j COMPLETED" >> logs/chain.log; break;;
      FAILED|CANCELLED*|TIMEOUT|OUT_OF_ME*|NODE_FAIL) echo "$(date +%H:%M) $j $st -> chain stopped" >> logs/chain.log; exit 1;;
    esac
    sleep 30
  done
done
echo "$(date +%H:%M) chain done" >> logs/chain.log
