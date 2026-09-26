#!/bin/bash
#SBATCH -J er_gpu2
#SBATCH -p mtech
#SBATCH -c 16
#SBATCH --mem=300G
#SBATCH --gres=gpu:2
#SBATCH -t 06:00:00
#SBATCH -o /csehome/m25csa019/amazon_ml/er/logs/%x_%j.out
# both cross-encoder halves in ONE job (one GPU each, in parallel), then the merge: counts as one job for the QOS.
# ARGS: ce.py arguments without half=; MERGE: arguments of the merge call (e.g. "src/ce.py merge=1 name=cef tag=b1ka")
source /csehome/m25csa019/amazon_ml/er/jobs/env.sh
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
nvidia-smi --query-gpu=name,memory.total --format=csv
IFS=',' read -ra G <<< "$CUDA_VISIBLE_DEVICES"
L=/csehome/m25csa019/amazon_ml/er/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
CUDA_VISIBLE_DEVICES=${G[0]} $PY $ARGS half=0 > ${L}_h0.log 2>&1 & p0=$!
CUDA_VISIBLE_DEVICES=${G[1]} $PY $ARGS half=1 > ${L}_h1.log 2>&1 & p1=$!
wait $p0 || { echo "half 0 failed"; exit 1; }
wait $p1 || { echo "half 1 failed"; exit 1; }
$PY $MERGE
