#!/bin/bash
# submit the bigger cross-encoder on the union (kx + ANN) band once the chain's step 3 (er_kas1) has completed
cd /csehome/m25csa019/amazon_ml/er
until [ "$(sacct -j 41118 -n -X --format=State | tr -d ' ')" = "COMPLETED" ]; do
  st=$(sacct -j 41118 -n -X --format=State | tr -d ' '); case "$st" in FAILED|CANCELLED*|TIMEOUT) echo "kas1 $st" >> logs/chain.log; exit 1;; esac
  sleep 60; done
while [ "$(squeue -u $USER -h | wc -l)" -ge 3 ]; do sleep 30; done
j=$(sbatch --parsable -J er_cebka -t 30:00:00 --export=ALL,ARGS="src/ce.py split=tsim tag=w1ka lo=0.003 hi=0.997 ntr=3000000 epochs=2 name=cebka txt=raw model=BAAI/bge-reranker-base lr=2e-5 bs=128 pbs=512" jobs/gpu.sbatch)
echo "$(date +%H:%M) submitted cebka $j" >> logs/chain.log
