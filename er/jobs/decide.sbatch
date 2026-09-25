#!/bin/bash
#SBATCH -J er_decide
#SBATCH -p mtech
#SBATCH -c 16
#SBATCH --mem=160G
#SBATCH -t 03:00:00
#SBATCH -o /csehome/m25csa019/amazon_ml/er/logs/%x_%j.out
source /csehome/m25csa019/amazon_ml/er/jobs/env.sh
set -e
OUT=/scratch/m25csa019/er_out/$TAG
$PY src/decide.py tune tag=$TAG
$PY src/decide.py apply tag=$TAG out=$OUT cand=cand_test
cd /csehome/m25csa019/amazon_ml/student_resource
$PY utils/validate_submission.py --matching $OUT/matching_results.tsv --candidate $OUT/candidate_pairs.tsv --test-dir dataset/test
