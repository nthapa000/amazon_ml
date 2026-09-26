#!/bin/bash
#SBATCH -J er_gpubw
#SBATCH -p mtech
#SBATCH -c 8
#SBATCH --mem=110G
#SBATCH --gres=gpu:1
#SBATCH -t 16:00:00
#SBATCH -o /csehome/m25csa019/amazon_ml/er/logs/%x_%j.out
# like gpu.sbatch, but only runs on an RTX PRO 6000 Blackwell: on any other GPU it exits with code 3 before loading data
# (cn07's gres is untyped: 2 Blackwell + 4 A6000), so jobs/bw_retry.sh can resubmit it.
source /csehome/m25csa019/amazon_ml/er/jobs/env.sh
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
G=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1); echo "GPU: $G"
case "$G" in *Blackwell*) ;; *) echo "not a Blackwell GPU, exiting"; exit 3;; esac
$PY $ARGS
