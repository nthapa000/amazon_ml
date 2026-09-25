#!/bin/bash
#SBATCH -J er_py
#SBATCH -p mtech
#SBATCH -c 32
#SBATCH --mem=200G
#SBATCH -t 04:00:00
#SBATCH -o /csehome/m25csa019/amazon_ml/er/logs/%x_%j.out
source /csehome/m25csa019/amazon_ml/er/jobs/env.sh
set -e
$PY $ARGS
