#!/bin/bash
# waits for the cef merge (jobs/cef2.dag), then runs jobs/fin4.dag (fin3 with cef instead of ceb + fold check vs fin3)
cd /csehome/m25csa019/amazon_ml/er
until grep -q -E "mg \([0-9]+\) COMPLETED" logs/dag_cef2.log; do
  if grep -q -E "(h1|mg) .*(FAILED|CANCELLED|SKIPPED|OUT_OF|TIMEOUT)" logs/dag_cef2.log; then
    echo "$(date +%H:%M) cef failed, fin4 not started" >> logs/dag_fin4.log; exit 1
  fi
  sleep 30
done
DAG_MAXJ=2 exec python3 jobs/dag.py jobs/fin4.dag logs/dag_fin4.log
