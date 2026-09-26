#!/bin/bash
#SBATCH -J er_gpu2bw
#SBATCH -p mtech
#SBATCH -c 16
#SBATCH --mem=200G
#SBATCH --gres=gpu:2
#SBATCH -t 19:00:00
#SBATCH -o /csehome/m25csa019/amazon_ml/er/logs/%x_%j.out
# both cross-encoder halves in ONE job (one Blackwell each, in parallel), then the merge: one QOS slot for ~12 h,
# which leaves the second running slot for CPU work. Exits 3 unless both allocated GPUs are RTX PRO 6000 Blackwell
# (cn07's gres is untyped) so that jobs/bw_retry.sh can resubmit. ARGS: ce.py args without half=; MERGE: merge call.
source /csehome/m25csa019/amazon_ml/er/jobs/env.sh
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
nvidia-smi --query-gpu=name,memory.total --format=csv
[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c Blackwell)" -eq 2 ] || { echo "not 2 Blackwell GPUs"; exit 3; }
IFS=',' read -ra G <<< "$CUDA_VISIBLE_DEVICES"
L=/csehome/m25csa019/amazon_ml/er/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
CUDA_VISIBLE_DEVICES=${G[0]} $PY $ARGS half=0 > ${L}_h0.log 2>&1 & p0=$!
CUDA_VISIBLE_DEVICES=${G[1]} $PY $ARGS half=1 > ${L}_h1.log 2>&1 & p1=$!
wait $p0 || { echo "half 0 failed"; exit 1; }
wait $p1 || { echo "half 1 failed"; exit 1; }
$PY $MERGE
