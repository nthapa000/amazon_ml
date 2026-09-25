#!/bin/bash
#SBATCH -J er_prep
#SBATCH -p mtech
#SBATCH -c 32
#SBATCH --mem=200G
#SBATCH -t 04:00:00
#SBATCH -o /csehome/m25csa019/amazon_ml/er/logs/%x_%j.out
source /csehome/m25csa019/amazon_ml/er/jobs/env.sh
set -e
[ -f $ER_WORK/gt_pairs.parquet ] || $PY src/prep.py raw
$PY src/dicts.py
$PY src/prep.py norm
