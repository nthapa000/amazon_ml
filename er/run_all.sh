#!/bin/bash
# End-to-end pipeline: raw TSVs -> $OUT/matching_results.tsv + $OUT/candidate_pairs.tsv.
# Generated from jobs/final.dag by jobs/gen_run_all.py (same commands, same order). Steps marked [GPU] need one
# CUDA GPU (A30 / A6000 / A100 / RTX PRO 6000; ceq and ceqf, the 4B reranker, were run on 97 GB RTX PRO 6000 Blackwell).
# Use the erbw environment (envs/er + torch 2.7.1+cu128 + peft) as $PY. CPU steps: 32 CPUs, ~230 GB RAM.
# On SLURM: python3 jobs/dag.py jobs/final.dag
set -e
source "$(dirname "$0")/jobs/env.sh"          # ER_DATA, ER_WORK, HF_HOME, PY
OUT=${OUT:-$ER_WORK/out}
VALIDATOR=${VALIDATOR:-$ER_DATA/../utils/validate_submission.py}
mkdir -p "$ER_WORK" "$OUT"

# ---- 1/22 base: normalize, test-like split, blocking (top-100 caps), pair features
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

# ---- 2/22 a1: stage-1 LightGBM on the K100 candidates   (after: base)
$PY src/train.py split=tsim tr=f_cand_tsim_k100 te=f_cand_test_k100 extra=xf_tsim_k100+f2_tsim_k100:xf_test_k100+f2_test_k100 tag=a1 frac=0.15 rounds=1000

# ---- 3/22 cev: cross-encoder 1 (English MiniLM, normalized text) on the a1 uncertain band  [GPU]   (after: a1)
$PY src/ce.py split=tsim tag=a1 lo=0.003 hi=0.997 ntr=3000000 epochs=2 name=cev model=cross-encoder/ms-marco-MiniLM-L12-v2

# ---- 4/22 kx: sibling expansion around confident a1 matches -> kx candidate set + features   (after: a1)
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

# ---- 5/22 ann: ANN pass (frozen multilingual-e5-small), validation split  [GPU]   (after: kx)
$PY src/ann.py split=tsim base=cand_tsim_kx k=3 out=ann_tsim

# ---- 6/22 annte: ANN pass, test split  [GPU]   (after: ann)
$PY src/ann.py split=test base=cand_test_kx k=3 out=ann_test

# ---- 7/22 b1: stage-1 LightGBM on the kx candidates   (after: kx)
$PY src/train.py split=tsim tr=f_cand_tsim_kx te=f_cand_test_kx extra=xf_tsim_kx+f2_tsim_kx+sib_tsim:xf_test_kx+f2_test_kx+sib_test tag=b1 frac=0.15 rounds=1000

# ---- 8/22 ka: merge ANN candidates (ka set), ANN stage-1 model, context + sibling-support features   (after: annte b1)
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

# ---- 9/22 cem: cross-encoder 2 (multilingual MiniLM, raw text) on the b1ka uncertain band  [GPU]   (after: ka)
$PY src/ce.py split=tsim tag=b1ka lo=0.003 hi=0.997 ntr=3000000 epochs=2 name=cem txt=raw model=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1

# ---- 10/22 ceb: cross-encoder 3 (bge-reranker-base, raw text) on the b1ka uncertain band  [GPU]   (after: ka)
$PY src/ce.py split=tsim tag=b1ka lo=0.003 hi=0.997 ntr=3000000 epochs=2 name=ceb txt=raw model=BAAI/bge-reranker-base lr=2e-5 bs=128 pbs=512

# ---- 11/22 opt: Optuna: 30 trials of stage-2 LightGBM parameters -> params_fin.json   (after: cev cem ceb)
$PY src/tune.py tr=f_cand_tsim_ka extra=xf_tsim_ka+f2_tsim_ka+sib_tsim_ka+annf_tsim_ka+ctx_b1ka_train+cev_a1_train+sup_ka_tsim+cem_b1ka_train+ceb_b1ka_train sub=oof_b1ka:0.001 trials=30 frac=0.4 out=fin

# ---- 12/22 cer: cross-encoder 4 (bge-reranker-v2-m3, raw text, 600k pairs) -> used only by the teacher fin2  [GPU]   (after: ka)
$PY src/ce.py split=tsim tag=b1ka lo=0.003 hi=0.997 txt=raw epochs=1 model=BAAI/bge-reranker-v2-m3 lr=2e-5 bs=64 pbs=512 save=0 ntr=600000 name=cer

# ---- 13/22 fin2: teacher stage 2 (tuned params + cer) for the French self-training set   (after: opt cer)
$PY src/train.py split=tsim tr=f_cand_tsim_ka te=f_cand_test_ka sub=oof_b1ka:pte_b1ka:0.001 frac=1.0 lr=0.05 seed=1 rounds=4000 pfile=params_fin.json extra=xf_tsim_ka+f2_tsim_ka+sib_tsim_ka+annf_tsim_ka+ctx_b1ka_train+cev_a1_train+sup_ka_tsim+cem_b1ka_train+ceb_b1ka_train+cer_b1ka_train:xf_test_ka+f2_test_ka+sib_test_ka+annf_test_ka+ctx_b1ka_test+cev_a1_test+sup_ka_test+cem_b1ka_test+ceb_b1ka_test+cer_b1ka_test tag=fin2
$PY src/decide.py tune tag=fin2 split=tsim

# ---- 14/22 fra: French self-training set: teacher pseudo-labels on unseen-country test pairs + exact S1-S1 negatives + noised S1 positives   (after: fin2)
$PY src/fradapt.py tag=fin2 out=fra_b1ka lw=0

# ---- 15/22 fra2: French self-training set, second version: the synthetic positives also get the generic last-word changes learned from the test records (used by ceqf)   (after: fin2)
$PY src/fradapt.py tag=fin2 out=fra_b1kb

# ---- 16/22 cef: cross-encoder 3b: bge-reranker-base warm start (ceb) + French self-training set + US/India replay  [GPU]   (after: fra)
$PY src/ce.py split=tsim tag=b1ka lo=0.003 hi=0.997 txt=raw epochs=1 model=BAAI/bge-reranker-base init=ceb_b1ka extra=fra_b1ka lr=1e-5 bs=128 pbs=512 ntr=600000 name=cef

# ---- 17/22 cer2: cross-encoder 5 (bge-reranker-v2-m3, full band recipe)  [GPU]   (after: ka)
$PY src/ce.py split=tsim tag=b1ka lo=0.003 hi=0.997 txt=raw ntr=3000000 epochs=2 model=BAAI/bge-reranker-v2-m3 lr=2e-5 bs=64 pbs=512 save=0 name=cer2

# ---- 18/22 ceq: cross-encoder 6 (Qwen3-Reranker-4B, LoRA r=16, bf16)  [GPU, large]   (after: ka)
$PY src/ce.py split=tsim tag=b1ka lo=0.003 hi=0.997 txt=raw ntr=1200000 epochs=2 model=Qwen/Qwen3-Reranker-4B bf16=1 lora=16 bs=32 pbs=128 ml=192 lr=1e-4 name=ceq

# ---- 19/22 fin5: stage 2 twice (with ceb / with cef; both + cer2 + ceq), logit blend tuned on OOF   (after: opt cef cer2 ceq)
$PY src/train.py split=tsim tr=f_cand_tsim_ka te=f_cand_test_ka sub=oof_b1ka:pte_b1ka:0.001 frac=1.0 lr=0.05 seed=1 rounds=4000 pfile=params_fin.json extra=xf_tsim_ka+f2_tsim_ka+sib_tsim_ka+annf_tsim_ka+ctx_b1ka_train+cev_a1_train+sup_ka_tsim+cem_b1ka_train+ceb_b1ka_train+cer2_b1ka_train+ceq_b1ka_train:xf_test_ka+f2_test_ka+sib_test_ka+annf_test_ka+ctx_b1ka_test+cev_a1_test+sup_ka_test+cem_b1ka_test+ceb_b1ka_test+cer2_b1ka_test+ceq_b1ka_test tag=fin5a
$PY src/decide.py tune tag=fin5a split=tsim
$PY src/train.py split=tsim tr=f_cand_tsim_ka te=f_cand_test_ka sub=oof_b1ka:pte_b1ka:0.001 frac=1.0 lr=0.05 seed=1 rounds=4000 pfile=params_fin.json extra=xf_tsim_ka+f2_tsim_ka+sib_tsim_ka+annf_tsim_ka+ctx_b1ka_train+cev_a1_train+sup_ka_tsim+cem_b1ka_train+cef_b1ka_train+cer2_b1ka_train+ceq_b1ka_train:xf_test_ka+f2_test_ka+sib_test_ka+annf_test_ka+ctx_b1ka_test+cev_a1_test+sup_ka_test+cem_b1ka_test+cef_b1ka_test+cer2_b1ka_test+ceq_b1ka_test tag=fin5b
$PY src/decide.py tune tag=fin5b split=tsim
$PY src/blend.py tags=fin5a+fin5b out=fin5 margin=0.3 grid=1
$PY src/decide.py tune tag=fin5 split=tsim

# ---- 20/22 ceqf: ceq warm start + French self-training set; re-scores only unseen-country test pairs  [GPU, large]   (after: ceq fra2)
$PY src/ce.py split=tsim tag=b1ka lo=0.003 hi=0.997 txt=raw ntr=150000 epochs=1 model=Qwen/Qwen3-Reranker-4B bf16=1 lora=16 bs=32 pbs=128 ml=192 lr=2.5e-5 init=ceq_b1ka extra=fra_b1kb nx=200000 unseen=1 name=ceqf half=0
$PY src/ce.py split=tsim tag=b1ka lo=0.003 hi=0.997 txt=raw ntr=150000 epochs=1 model=Qwen/Qwen3-Reranker-4B bf16=1 lora=16 bs=32 pbs=128 ml=192 lr=2.5e-5 init=ceq_b1ka extra=fra_b1kb nx=200000 unseen=1 name=ceqf half=1
$PY src/ce.py merge=1 unseen=1 base=ceq name=ceqf tag=b1ka

# ---- 21/22 fin6: fin5 fold models re-score the test pairs with ceqf patched in (no retraining)   (after: fin5 ceqf)
$PY src/train.py split=tsim te=f_cand_test_ka sub=oof_b1ka:pte_b1ka:0.001 extra=x:xf_test_ka+f2_test_ka+sib_test_ka+annf_test_ka+ctx_b1ka_test+cev_a1_test+sup_ka_test+cem_b1ka_test+ceb_b1ka_test+cer2_b1ka_test+ceqf_b1ka_test alias=ceqf:ceq models=fin5a tag=fin6a
$PY src/train.py split=tsim te=f_cand_test_ka sub=oof_b1ka:pte_b1ka:0.001 extra=x:xf_test_ka+f2_test_ka+sib_test_ka+annf_test_ka+ctx_b1ka_test+cev_a1_test+sup_ka_test+cem_b1ka_test+cef_b1ka_test+cer2_b1ka_test+ceqf_b1ka_test alias=ceqf:ceq models=fin5b tag=fin6b
$PY src/blend.py tags=fin6a+fin6b out=fin6 margin=0.3 grid=1
$PY src/decide.py tune tag=fin6 split=tsim

# ---- 22/22 out: final candidate set = stage-1 filter p1 >= 0.005, decision rule tuned on OOF, unseen-country shift, write outputs, validate   (after: fin6)
$PY src/cands.py s1=b1ka fin=fin6 thr=0.005 out=final
$PY src/decide.py tune tag=final split=tsim
$PY src/decide.py apply tag=final out=$OUT cand=cand_test_final fr=-2
$PY $VALIDATOR --matching $OUT/matching_results.tsv --candidate $OUT/candidate_pairs.tsv --test-dir $ER_DATA/test
