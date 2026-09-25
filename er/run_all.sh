#!/bin/bash
# End-to-end pipeline: raw TSVs -> output/matching_results.tsv + output/candidate_pairs.tsv
# Run inside a SLURM allocation (32 CPUs, ~230 GB RAM) or via jobs/*.sbatch. Env vars: see jobs/env.sh
set -e
source "$(dirname "$0")/jobs/env.sh"
OUT=${OUT:-/scratch/m25csa019/er_out/final}

$PY src/prep.py raw                      # TSV -> parquet, GT pairs
$PY src/dicts.py                         # learned native-script -> latin token dict (train pairs only)
$PY src/prep.py norm                     # normalize names / addresses
for S in train test; do
  $PY src/block.py gen $S                # blocking -> cand_$S
  $PY src/feats.py split=$S cand=cand_$S # pair features
  $PY src/xf.py split=$S cand=cand_$S    # split-level statistics features
done
# stage 1
$PY src/train.py tr=f_cand_train te=f_cand_test extra=xf_train:xf_test tag=s1 frac=0.2
$PY src/ctx.py tag=s1
# stage 2 (context / competition features from stage-1 OOF)
$PY src/train.py tr=f_cand_train te=f_cand_test extra=xf_train+ctx_s1_train:xf_test+ctx_s1_test tag=s2 frac=0.2
$PY src/decide.py tune tag=s2
$PY src/decide.py apply tag=s2 out=$OUT cand=cand_test
cd /csehome/m25csa019/amazon_ml/student_resource
$PY utils/validate_submission.py --matching $OUT/matching_results.tsv --candidate $OUT/candidate_pairs.tsv --test-dir dataset/test
