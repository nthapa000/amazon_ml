#!/bin/bash
# End-to-end pipeline: raw TSVs -> $OUT/matching_results.tsv + $OUT/candidate_pairs.tsv.
# Generated from jobs/final.dag by jobs/gen_run_all.py (same commands, same order). Steps marked [GPU] need one
# CUDA GPU (sm_80/sm_86: A30, A6000, A100). CPU steps: 32 CPUs, ~230 GB RAM. On SLURM: python3 jobs/dag.py jobs/final.dag
set -e
source "$(dirname "$0")/jobs/env.sh"          # ER_DATA, ER_WORK, HF_HOME, PY
OUT=${OUT:-$ER_WORK/out}
VALIDATOR=${VALIDATOR:-$ER_DATA/../utils/validate_submission.py}
mkdir -p "$ER_WORK" "$OUT"

# ---- 1/11 base: normalize, test-like split, blocking (top-100 caps), pair features
$PY src/prep.py raw
$PY src/dicts.py
$PY src/prep.py norm
$PY src/mksim.py drop=0.19
$PY src/block.py gen tsim
$PY src/block.py gen test
$PY src/ext.py make split=tsim k1=100 k2=10
$PY src/ext.py make split=test k1=100 k2=10
$PY src/feats.py split=tsim cand=cand_tsim_k100 parts=24
$PY src/feats.py split=test cand=cand_test_k100 parts=24
$PY src/xf.py split=tsim cand=cand_tsim_k100 out=xf_tsim_k100
$PY src/xf.py split=test cand=cand_test_k100 out=xf_test_k100
$PY src/feats2.py split=tsim cand=cand_tsim_k100 out=f2_tsim_k100
$PY src/feats2.py split=test cand=cand_test_k100 out=f2_test_k100

# ---- 2/11 a1: stage-1 LightGBM on the K100 candidates   (after: base)
$PY src/train.py split=tsim tr=f_cand_tsim_k100 te=f_cand_test_k100 extra=xf_tsim_k100+f2_tsim_k100:xf_test_k100+f2_test_k100 tag=a1 frac=0.15 rounds=1000

# ---- 3/11 cev: cross-encoder 1 (English MiniLM, normalized text) on the a1 uncertain band  [GPU]   (after: a1)
$PY src/ce.py split=tsim tag=a1 lo=0.003 hi=0.997 ntr=3000000 epochs=2 name=cev model=cross-encoder/ms-marco-MiniLM-L12-v2

# ---- 4/11 kx: sibling expansion around confident a1 matches -> kx candidate set + features   (after: a1)
$PY src/expand.py split=tsim pred=oof_a1 thr=0.9 base=cand_tsim_k100 k1=20 k2=5
$PY src/expand.py split=test pred=pte_a1 thr=0.9 base=cand_test_k100 k1=20 k2=5
$PY src/kx.py prep split=tsim
$PY src/kx.py prep split=test
$PY src/feats.py split=tsim cand=expc_tsim parts=8
$PY src/feats.py split=test cand=expc_test parts=8
$PY src/kx.py link split=tsim
$PY src/kx.py link split=test
$PY src/xf.py split=tsim cand=cand_tsim_kx out=xf_tsim_kx
$PY src/xf.py split=test cand=cand_test_kx out=xf_test_kx
$PY src/feats2.py split=tsim cand=cand_tsim_kx out=f2_tsim_kx
$PY src/feats2.py split=test cand=cand_test_kx out=f2_test_kx

# ---- 5/11 ann: ANN pass (frozen multilingual-e5-small), validation split  [GPU]   (after: kx)
$PY src/ann.py split=tsim base=cand_tsim_kx k=3 out=ann_tsim

# ---- 6/11 annte: ANN pass, test split  [GPU]   (after: ann)
$PY src/ann.py split=test base=cand_test_kx k=3 out=ann_test

# ---- 7/11 b1: stage-1 LightGBM on the kx candidates   (after: kx)
$PY src/train.py split=tsim tr=f_cand_tsim_kx te=f_cand_test_kx extra=xf_tsim_kx+f2_tsim_kx+sib_tsim:xf_test_kx+f2_test_kx+sib_test tag=b1 frac=0.15 rounds=1000

# ---- 8/11 ka: merge ANN candidates (ka set), ANN stage-1 model, context + sibling-support features   (after: annte b1)
$PY src/annka.py cand split=tsim k=3
$PY src/annka.py cand split=test k=3
$PY src/feats.py split=tsim cand=cand_ann_tsim parts=8
$PY src/feats.py split=test cand=cand_ann_test parts=8
$PY src/xf.py split=tsim cand=cand_ann_tsim out=xf_ann_tsim
$PY src/xf.py split=test cand=cand_ann_test out=xf_ann_test
$PY src/feats2.py split=tsim cand=cand_ann_tsim out=f2_ann_tsim
$PY src/feats2.py split=test cand=cand_ann_test out=f2_ann_test
$PY src/annka.py union split=tsim pred=oof_b1 thr=0.0005
$PY src/annka.py union split=test pred=pte_b1 thr=0.0005
$PY src/train.py split=tsim tr=f_cand_ann_tsim te=f_cand_ann_test extra=xf_ann_tsim+f2_ann_tsim+annf_ann_tsim:xf_ann_test+f2_ann_test+annf_ann_test tag=b1ann frac=1.0 lr=0.05 seed=1 rounds=3000
$PY src/annka.py s1 tsim=b1 ann=b1ann out=b1ka
$PY src/ctx.py tag=b1ka thr=0.0005
$PY src/sup.py split=tsim pred=oof_b1ka thr=0.001 sib_thr=0.9 out=sup_ka_tsim
$PY src/sup.py split=test pred=pte_b1ka thr=0.001 sib_thr=0.9 out=sup_ka_test

# ---- 9/11 cem: cross-encoder 2 (multilingual MiniLM, raw text) on the b1ka uncertain band  [GPU]   (after: ka)
$PY src/ce.py split=tsim tag=b1ka lo=0.003 hi=0.997 ntr=3000000 epochs=2 name=cem txt=raw model=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1

# ---- 10/11 ceb: cross-encoder 3 (bge-reranker-base, raw text) on the b1ka uncertain band  [GPU]   (after: ka)
$PY src/ce.py split=tsim tag=b1ka lo=0.003 hi=0.997 ntr=3000000 epochs=2 name=ceb txt=raw model=BAAI/bge-reranker-base lr=2e-5 bs=128 pbs=512

# ---- 11/11 fin: cascade stage-2 LightGBM, final candidate set = stage-1 filter p1 >= 0.005, decision rule tuned on OOF, write outputs, validate   (after: cev cem ceb)
$PY src/train.py split=tsim tr=f_cand_tsim_ka te=f_cand_test_ka sub=oof_b1ka:pte_b1ka:0.001 frac=1.0 lr=0.05 seed=1 rounds=4000 extra=xf_tsim_ka+f2_tsim_ka+sib_tsim_ka+annf_tsim_ka+ctx_b1ka_train+cev_a1_train+sup_ka_tsim+cem_b1ka_train+ceb_b1ka_train:xf_test_ka+f2_test_ka+sib_test_ka+annf_test_ka+ctx_b1ka_test+cev_a1_test+sup_ka_test+cem_b1ka_test+ceb_b1ka_test tag=fin
$PY src/cands.py s1=b1ka fin=fin thr=0.005 out=finc
$PY src/decide.py tune tag=finc split=tsim
$PY src/decide.py apply tag=finc out=$OUT cand=cand_test_final
$PY $VALIDATOR --matching $OUT/matching_results.tsv --candidate $OUT/candidate_pairs.tsv --test-dir $ER_DATA/test
