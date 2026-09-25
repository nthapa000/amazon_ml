#!/bin/bash
#SBATCH -J er_gpu
#SBATCH -p mtech
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH -t 10:00:00
#SBATCH -o /csehome/m25csa019/amazon_ml/er/logs/%x_%j.out
source /csehome/m25csa019/amazon_ml/er/jobs/env.sh
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
nvidia-smi --query-gpu=name,memory.total --format=csv
$PY $ARGS
