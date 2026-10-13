#!/bin/bash
# End-to-end pipeline: raw challenge TSVs -> $OUT/matching_results.tsv + $OUT/candidate_pairs.tsv
#
#   ER_DATA=/path/to/dataset bash src/run_all.sh          (run from code/business_entity_resolution/)
#
# Environment (all optional except ER_DATA):
#   ER_DATA        dataset directory containing train/ and test/ (the challenge files, unchanged)
#   ER_WORK        intermediate files, ~200 GB at peak              (default: ./work)
#   OUT            where the two output files are written          (default: ./output)
#   ER_PRETRAINED  pretrained models from src/download_models.py   (default: ./models)
#   PY             Python interpreter of the environment in requirements.txt (default: python)
#   VALIDATOR      official validator                             (default: $ER_DATA/../utils/validate_submission.py)
#   FROM           resume from this step number                   (default: 1)
#   TEAM_SCORES    directory with the teammates' stage-2 score files (step 19): oof_kcxQ.parquet, pte_kcxQ.parquet,
#                  oof_kfx.parquet, pte_kfx.parquet                 (default: ./team_scores)
# Steps marked [GPU] need one CUDA GPU; ceq (step 16) ran on a 97 GB GPU. CPU steps need ~230 GB RAM.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${ER_DATA:?set ER_DATA to the dataset directory that contains train/ and test/}"
export ER_DATA=$(cd "$ER_DATA" && pwd)
export ER_WORK=${ER_WORK:-$PWD/work}
OUT=${OUT:-$PWD/output}
M=${ER_PRETRAINED:-$PWD/models}
PY=${PY:-python}
VALIDATOR=${VALIDATOR:-$ER_DATA/../utils/validate_submission.py}
FROM=${FROM:-1}
TEAM_SCORES=${TEAM_SCORES:-$PWD/team_scores}
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-$(nproc)}   # worker / thread count read by the Python scripts
export POLARS_MAX_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false   # no network access after download
export PYTHONWARNINGS=ignore::DeprecationWarning   # pinned polars 1.44 announces its 2.0 API changes; results unaffected
mkdir -p "$ER_WORK" "$OUT"
for m in multilingual-e5-small ms-marco-MiniLM-L12-v2 mmarco-mMiniLMv2-L12-H384-v1 bge-reranker-base bge-reranker-v2-m3 \
         Qwen3-Reranker-4B; do
  [ -f "$M/$m/config.json" ] || { echo "missing pretrained model $M/$m: run '$PY src/download_models.py out=$M' first"; exit 1; }
done

STEP=0
step() { STEP=$1; shift; [ "$STEP" -ge "$FROM" ] && echo "==== step $STEP/21: $*" || true; }
run() { [ "$STEP" -ge "$FROM" ] || return 0; echo "+ $PY $*"; "$PY" "$@"; }

step 1 "normalization, test-like validation split, compound-key blocking (top-100 caps), pair features"
run src/prep.py raw
run src/dicts.py
run src/prep.py norm
run src/mksim.py drop=0.19
run src/block.py gen tsim
run src/block.py gen test
run src/ext.py make split=tsim k1=100 k2=10
run src/ext.py make split=test k1=100 k2=10
run src/feats.py split=tsim cand=cand_tsim_k100 parts=24
run src/feats.py split=test cand=cand_test_k100 parts=24
run src/xf.py split=tsim cand=cand_tsim_k100 out=xf_tsim_k100
run src/xf.py split=test cand=cand_test_k100 out=xf_test_k100
run src/feats2.py split=tsim cand=cand_tsim_k100 out=f2_tsim_k100
run src/feats2.py split=test cand=cand_test_k100 out=f2_test_k100

step 2 "stage-1 LightGBM a1 on the top-100 key candidates"
run src/train.py split=tsim tr=f_cand_tsim_k100 te=f_cand_test_k100 extra=xf_tsim_k100+f2_tsim_k100:xf_test_k100+f2_test_k100 tag=a1 frac=0.15 rounds=1000

step 3 "[GPU] cross-encoder cev (ms-marco-MiniLM-L12-v2, normalized text) on the a1 uncertain band"
run src/ce.py split=tsim tag=a1 lo=0.003 hi=0.997 ntr=3000000 epochs=2 name=cev model=$M/ms-marco-MiniLM-L12-v2

step 4 "sibling expansion around confident a1 matches -> kx candidate set + its features"
run src/expand.py split=tsim pred=oof_a1 thr=0.9 base=cand_tsim_k100 k1=20 k2=5
run src/expand.py split=test pred=pte_a1 thr=0.9 base=cand_test_k100 k1=20 k2=5
run src/kx.py prep split=tsim
run src/kx.py prep split=test
run src/feats.py split=tsim cand=expc_tsim parts=8
run src/feats.py split=test cand=expc_test parts=8
run src/kx.py link split=tsim
run src/kx.py link split=test
run src/xf.py split=tsim cand=cand_tsim_kx out=xf_tsim_kx
run src/xf.py split=test cand=cand_test_kx out=xf_test_kx
run src/feats2.py split=tsim cand=cand_tsim_kx out=f2_tsim_kx
run src/feats2.py split=test cand=cand_test_kx out=f2_test_kx

step 5 "[GPU] ANN pass (frozen multilingual-e5-small): top-3 S1 per S2/S3 record within the country"
run src/ann.py split=tsim base=cand_tsim_kx k=3 out=ann_tsim model=$M/multilingual-e5-small
run src/ann.py split=test base=cand_test_kx k=3 out=ann_test model=$M/multilingual-e5-small

step 6 "stage-1 LightGBM b1 on the kx candidates"
run src/train.py split=tsim tr=f_cand_tsim_kx te=f_cand_test_kx extra=xf_tsim_kx+f2_tsim_kx+sib_tsim:xf_test_kx+f2_test_kx+sib_test tag=b1 frac=0.15 rounds=1000

step 7 "ANN candidates: features, union with kx (ka set), stage-1 model b1ann, competition + sibling-support features"
run src/annka.py cand split=tsim k=3
run src/annka.py cand split=test k=3
run src/feats.py split=tsim cand=cand_ann_tsim parts=8
run src/feats.py split=test cand=cand_ann_test parts=8
run src/xf.py split=tsim cand=cand_ann_tsim out=xf_ann_tsim
run src/xf.py split=test cand=cand_ann_test out=xf_ann_test
run src/feats2.py split=tsim cand=cand_ann_tsim out=f2_ann_tsim
run src/feats2.py split=test cand=cand_ann_test out=f2_ann_test
run src/annka.py union split=tsim pred=oof_b1 thr=0.0005
run src/annka.py union split=test pred=pte_b1 thr=0.0005
run src/train.py split=tsim tr=f_cand_ann_tsim te=f_cand_ann_test extra=xf_ann_tsim+f2_ann_tsim+annf_ann_tsim:xf_ann_test+f2_ann_test+annf_ann_test tag=b1ann frac=1.0 lr=0.05 seed=1 rounds=3000
run src/annka.py s1 tsim=b1 ann=b1ann out=b1ka
run src/ctx.py tag=b1ka thr=0.0005
run src/sup.py split=tsim pred=oof_b1ka thr=0.001 sib_thr=0.9 out=sup_ka_tsim
run src/sup.py split=test pred=pte_b1ka thr=0.001 sib_thr=0.9 out=sup_ka_test

step 8 "[GPU] cross-encoder cem (mmarco-mMiniLMv2, raw text) on the b1ka uncertain band"
run src/ce.py split=tsim tag=b1ka lo=0.003 hi=0.997 ntr=3000000 epochs=2 name=cem txt=raw model=$M/mmarco-mMiniLMv2-L12-H384-v1

step 9 "[GPU] cross-encoder ceb (bge-reranker-base, raw text) on the b1ka uncertain band"
run src/ce.py split=tsim tag=b1ka lo=0.003 hi=0.997 ntr=3000000 epochs=2 name=ceb txt=raw model=$M/bge-reranker-base lr=2e-5 bs=128 pbs=512

step 10 "Optuna: 30 trials of stage-2 LightGBM parameters -> params_fin.json"
run src/tune.py tr=f_cand_tsim_ka extra=xf_tsim_ka+f2_tsim_ka+sib_tsim_ka+annf_tsim_ka+ctx_b1ka_train+cev_a1_train+sup_ka_tsim+cem_b1ka_train+ceb_b1ka_train sub=oof_b1ka:0.001 trials=30 frac=0.4 out=fin

step 11 "[GPU] cross-encoder cer (bge-reranker-v2-m3, 600k pairs), input of the teacher model only"
run src/ce.py split=tsim tag=b1ka lo=0.003 hi=0.997 txt=raw epochs=1 model=$M/bge-reranker-v2-m3 lr=2e-5 bs=64 pbs=512 save=0 ntr=600000 name=cer

step 12 "teacher stage 2 (fin2) for the self-training set"
run src/train.py split=tsim tr=f_cand_tsim_ka te=f_cand_test_ka sub=oof_b1ka:pte_b1ka:0.001 frac=1.0 lr=0.05 seed=1 rounds=4000 pfile=params_fin.json extra=xf_tsim_ka+f2_tsim_ka+sib_tsim_ka+annf_tsim_ka+ctx_b1ka_train+cev_a1_train+sup_ka_tsim+cem_b1ka_train+ceb_b1ka_train+cer_b1ka_train:xf_test_ka+f2_test_ka+sib_test_ka+annf_test_ka+ctx_b1ka_test+cev_a1_test+sup_ka_test+cem_b1ka_test+ceb_b1ka_test+cer_b1ka_test tag=fin2
run src/decide.py tune tag=fin2 split=tsim

step 13 "self-training set for the unseen country: teacher pseudo-labels + exact S1-S1 negatives + noised S1 positives"
run src/fradapt.py tag=fin2 out=fra_b1ka

step 14 "[GPU] cross-encoder cef: ceb continued on the self-training set + US/India replay"
run src/ce.py split=tsim tag=b1ka lo=0.003 hi=0.997 txt=raw epochs=1 model=$M/bge-reranker-base init=ceb_b1ka extra=fra_b1ka lr=1e-5 bs=128 pbs=512 ntr=600000 name=cef

step 15 "[GPU] cross-encoder cer2 (bge-reranker-v2-m3, full band, 2 epochs)"
run src/ce.py split=tsim tag=b1ka lo=0.003 hi=0.997 txt=raw ntr=3000000 epochs=2 model=$M/bge-reranker-v2-m3 lr=2e-5 bs=64 pbs=512 save=0 name=cer2

step 16 "[GPU; we used a 97 GB GPU] cross-encoder ceq (Qwen3-Reranker-4B, LoRA r=16, bf16)"
run src/ce.py split=tsim tag=b1ka lo=0.003 hi=0.997 txt=raw ntr=1200000 epochs=2 model=$M/Qwen3-Reranker-4B bf16=1 lora=16 bs=32 pbs=128 ml=192 lr=1e-4 name=ceq

step 17 "final matching model: stage-2 LightGBM with ceb (fin5a) and with cef (fin5b), logit blend, decision rule"
run src/train.py split=tsim tr=f_cand_tsim_ka te=f_cand_test_ka sub=oof_b1ka:pte_b1ka:0.001 frac=1.0 lr=0.05 seed=1 rounds=4000 pfile=params_fin.json extra=xf_tsim_ka+f2_tsim_ka+sib_tsim_ka+annf_tsim_ka+ctx_b1ka_train+cev_a1_train+sup_ka_tsim+cem_b1ka_train+ceb_b1ka_train+cer2_b1ka_train+ceq_b1ka_train:xf_test_ka+f2_test_ka+sib_test_ka+annf_test_ka+ctx_b1ka_test+cev_a1_test+sup_ka_test+cem_b1ka_test+ceb_b1ka_test+cer2_b1ka_test+ceq_b1ka_test tag=fin5a
run src/train.py split=tsim tr=f_cand_tsim_ka te=f_cand_test_ka sub=oof_b1ka:pte_b1ka:0.001 frac=1.0 lr=0.05 seed=1 rounds=4000 pfile=params_fin.json extra=xf_tsim_ka+f2_tsim_ka+sib_tsim_ka+annf_tsim_ka+ctx_b1ka_train+cev_a1_train+sup_ka_tsim+cem_b1ka_train+cef_b1ka_train+cer2_b1ka_train+ceq_b1ka_train:xf_test_ka+f2_test_ka+sib_test_ka+annf_test_ka+ctx_b1ka_test+cev_a1_test+sup_ka_test+cem_b1ka_test+cef_b1ka_test+cer2_b1ka_test+ceq_b1ka_test tag=fin5b
run src/blend.py tags=fin5a+fin5b out=fin5 margin=0.3 grid=1
run src/decide.py tune tag=fin5 split=tsim

step 18 "our team member fin5s: stage 2 fin3 -> sibling support sup3 from its scores -> fin5sa -> logit blend with fin5b"
run src/train.py split=tsim tr=f_cand_tsim_ka te=f_cand_test_ka sub=oof_b1ka:pte_b1ka:0.001 frac=1.0 lr=0.05 seed=1 rounds=4000 extra=xf_tsim_ka+f2_tsim_ka+sib_tsim_ka+annf_tsim_ka+ctx_b1ka_train+cev_a1_train+sup_ka_tsim+cem_b1ka_train+ceb_b1ka_train+cer2_b1ka_train:xf_test_ka+f2_test_ka+sib_test_ka+annf_test_ka+ctx_b1ka_test+cev_a1_test+sup_ka_test+cem_b1ka_test+ceb_b1ka_test+cer2_b1ka_test tag=fin3 pfile=params_fin.json
run src/sup.py split=tsim pred=oof_fin3 thr=0 sib_thr=0.9 out=sup3_ka_tsim
run src/sup.py split=test pred=pte_fin3 thr=0 sib_thr=0.9 out=sup3_ka_test
run src/train.py split=tsim tr=f_cand_tsim_ka te=f_cand_test_ka sub=oof_b1ka:pte_b1ka:0.001 frac=1.0 lr=0.05 seed=1 rounds=4000 pfile=params_fin.json extra=xf_tsim_ka+f2_tsim_ka+sib_tsim_ka+annf_tsim_ka+ctx_b1ka_train+cev_a1_train+sup3_ka_tsim+cem_b1ka_train+ceb_b1ka_train+cer2_b1ka_train+ceq_b1ka_train:xf_test_ka+f2_test_ka+sib_test_ka+annf_test_ka+ctx_b1ka_test+cev_a1_test+sup3_ka_test+cem_b1ka_test+ceb_b1ka_test+cer2_b1ka_test+ceq_b1ka_test tag=fin5sa
run src/blend.py tags=fin5sa+fin5b out=fin5s margin=0.3 grid=1

step 19 "teammates' members cxQ and fx: copy their stage-2 scores on the same pairs into ER_WORK and gate them"
if [ "$STEP" -ge "$FROM" ]; then
  for t in kcxQ kfx; do
    for k in oof pte; do
      [ -f "$TEAM_SCORES/${k}_$t.parquet" ] || { echo "missing $TEAM_SCORES/${k}_$t.parquet (see README, section 5)"; exit 1; }
      cp "$TEAM_SCORES/${k}_$t.parquet" "$ER_WORK/"
    done
  done
fi
run src/kcheck.py tag=kcxQ ref=fin5s others=kfx
run src/kcheck.py tag=kfx ref=fin5s others=kcxQ

step 20 "team blend tb5: logit average; US/India weights searched on tsim OOF (-> 0.7 cxQ / 0.3 fin5s / 0 fx), France equal thirds"
run src/team_blend.py tags=kcxQ+fin5s+kfx out=tb5 step=0.25 fine=0.05 unseen=equal

step 21 "outputs: decision rule + unseen-country shift fr=-2 -> matching_results.tsv; the scored pairs -> candidate_pairs.tsv"
run src/decide.py apply tag=tb5 "out=$OUT" fr=-2
if [ "$STEP" -ge "$FROM" ]; then
  if [ -f "$VALIDATOR" ]; then
    "$PY" "$VALIDATOR" --matching "$OUT/matching_results.tsv" --candidate "$OUT/candidate_pairs.tsv" --test-dir "$ER_DATA/test"
  else
    echo "official validator not found at $VALIDATOR (set VALIDATOR=...); outputs written to $OUT"
  fi
fi
