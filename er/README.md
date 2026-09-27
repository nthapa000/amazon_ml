# Business Entity Resolution — reproducible pipeline

Links every Source-1 business to its Source-2 / Source-3 records using only name, address and country, and writes
the two submission files:

- `matching_results.tsv`: one row per Source-1 entity, its matched S2/S3 ids (may be empty)
- `candidate_pairs.tsv`: the final candidate set the matching model runs on (~4.5 candidates per Source-1 entity)

**Pipeline in one line:** normalization → multi-pass candidate generation (compound keys, then sibling expansion,
then ANN; ~103 per S1) → pair features → stage-1 LightGBM, used as a learned candidate filter (p1 ≥ 0.005 → 4.47 per S1)
→ cross-encoders on the uncertain band (MiniLM, multilingual MiniLM, bge-reranker-base, bge-reranker-v2-m3,
Qwen3-Reranker-4B, plus French-adapted versions) → cascade stage-2 LightGBM (two variants, logit blend) → one-to-one
assignment with an expected-F0.5 set rule tuned on out-of-fold predictions → calibration shift for the country unseen in
training.

No external data, APIs or lookups are used. Dictionaries are learned from the training pairs, and France rules are
hand-written. France has no training labels, so two cross-encoders are continued on a self-training set built only from
the provided records: our own model's confident predictions on French test pairs, exact negatives between different
Source-1 records, and noised copies of Source-1 records (organizers' Q&A: self-training and synthetic pairs from the
provided records are allowed).

## 1. Environment

```
conda create -p <env> python=3.11
<env>/bin/pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
<env>/bin/pip install -r requirements.txt
```

**Pretrained weights**
- Download these once into `$HF_HOME`. GPU jobs run with `HF_HUB_OFFLINE=1`, so they must already be cached.
- Every model is used as published, with no business data added.

| model | licence | params | revision (HF commit) | used in |
|---|---|---|---|---|
| `intfloat/multilingual-e5-small` | MIT | 118M | `614241f622f53c4eeff9890bdc4f31cfecc418b3` | ANN pass (frozen, never fine-tuned) |
| `cross-encoder/ms-marco-MiniLM-L12-v2` | Apache-2.0 | 33M | `7b0235231ca2674cb8ca8f022859a6eba2b1c968` | cross-encoder `cev` |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Apache-2.0 | 118M | `1427fd652930e4ba29e8149678df786c240d8825` | cross-encoder `cem` |
| `BAAI/bge-reranker-base` | MIT | 278M | `2cfc18c9415c912f9d8155881c133215df768a70` | cross-encoders `ceb`, `cef` |
| `BAAI/bge-reranker-v2-m3` | Apache-2.0 | 568M | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` | cross-encoders `cer` (teacher only), `cer2` |
| `Qwen/Qwen3-Reranker-4B` | Apache-2.0 | 4.0B | `22e683669bc0f0bd69640a1354a6d0aebcfeede5` | cross-encoders `ceq`, `ceqf` (LoRA) |

Every model is at most 4B parameters (limit 8B), and every licence is MIT or Apache-2.0.

```
HF_HOME=<cache> <env>/bin/python -c "from huggingface_hub import snapshot_download as d; [d(m) for m in ['intfloat/multilingual-e5-small','cross-encoder/ms-marco-MiniLM-L12-v2','cross-encoder/mmarco-mMiniLMv2-L12-H384-v1','BAAI/bge-reranker-base','BAAI/bge-reranker-v2-m3','Qwen/Qwen3-Reranker-4B']]"
```

## 2. Paths

All four paths in `jobs/env.sh` can be overridden from the environment:

- `ER_DATA`: the dataset directory containing `train/` and `test/`
- `ER_WORK`: scratch space for intermediates, about 200 GB at peak
- `HF_HOME`: the weights cache
- `PY`: the Python interpreter

The output directory is `OUT` (default `$ER_WORK/out`).

## 3. Run end-to-end

**Option A, one machine:**
- Hardware: 32 CPUs, about 230 GB RAM, and one CUDA GPU.
- The PyTorch build above (2.7.1, CUDA 12.8) supports A30, A6000, A100 and Blackwell (sm_120).
- The 4B reranker steps (`ceq`, `ceqf`) ran on 97 GB RTX PRO 6000 Blackwell GPUs (LoRA, bf16, batch 32, 192 tokens).
  On one GPU, `ceq` takes about 20 h of training plus about 10 h of scoring. The SLURM run used one GPU per fold in parallel.

```
ER_DATA=<dataset> ER_WORK=<scratch> HF_HOME=<cache> PY=<env>/bin/python OUT=<outdir> bash run_all.sh
```

**Option B, SLURM (how the submitted files were produced):**
- `jobs/final.dag` lists the 22 jobs and their dependencies.
- `jobs/dag.py` submits each job when its inputs are ready, respecting the per-user job limit.
- CPU jobs use `jobs/multi.sbatch`. GPU jobs use `jobs/gpu.sbatch` (steps 3–10) or `jobs/multi.sbatch` with one GPU (steps 12–21).

```
python3 jobs/dag.py jobs/final.dag            # progress in logs/dag.log, job logs in logs/fz_<step>_<jobid>.out
```

`run_all.sh` is generated from `jobs/final.dag` and runs the same commands in the same order.

## 4. Steps

| # | step | what it does | main code | resources | time (A30 / 32 CPU) |
|---|---|---|---|---|---|
| 1 | base | TSV → parquet, learned transliteration dictionary (train pairs), normalization, test-like validation split `tsim` (19% of train S1 dropped), compound-key blocking with top-100 caps, pair features | `prep.py`, `dicts.py`, `norm.py`, `mksim.py`, `block.py`, `ext.py`, `feats.py`, `xf.py`, `feats2.py` | CPU | ~1 h 15 |
| 2 | a1 | stage-1 LightGBM on the K100 candidates (5-fold by S1) | `train.py` | CPU | ~1 h |
| 3 | cev | cross-encoder 1 (English MiniLM, normalized text) on a1's uncertain band, 2-fold by S1 | `ce.py` | GPU | ~1 h 15 |
| 4 | kx | sibling expansion around confident a1 matches (p ≥ 0.9, one-to-one), features for new pairs | `expand.py`, `kx.py`, `feats.py`, `xf.py`, `feats2.py` | CPU | ~25 min |
| 5 | ann | ANN pass on `tsim`: top-3 S1 per S2/S3 record within country | `ann.py` | GPU | ~35 min |
| 6 | annte | ANN pass on test | `ann.py` | GPU | ~50 min |
| 7 | b1 | stage-1 LightGBM on the kx candidates | `train.py` | CPU | ~1 h 45 |
| 8 | ka | ANN candidates: features, union with kx (`ka` set), own stage-1 model, context and sibling-support features | `annka.py`, `feats*.py`, `xf.py`, `train.py`, `ctx.py`, `sup.py` | CPU | ~35 min |
| 9 | cem | cross-encoder 2 (multilingual MiniLM, raw text) on the b1ka band | `ce.py` | GPU | ~1 h 25 |
| 10 | ceb | cross-encoder 3 (bge-reranker-base, raw text) on the b1ka band | `ce.py` | GPU | ~5 h |
| 11 | opt | Optuna: 30 trials of stage-2 LightGBM parameters on 2 folds (logloss) → `params_fin.json` | `tune.py` | CPU | ~20 min |
| 12 | cer | cross-encoder 4 (bge-reranker-v2-m3, 600k pairs, 1 epoch); only feeds the teacher `fin2` | `ce.py` | GPU | ~4 h 15 |
| 13 | fin2 | teacher stage 2 (tuned parameters + `cer`) | `train.py`, `decide.py` | CPU | ~35 min |
| 14 | fra | French self-training set from the teacher (original noise, `lw=0`), used by `cef` | `fradapt.py` | CPU | ~1 min |
| 15 | fra2 | the same set with generic last-word changes in the synthetic positives, used by `ceqf` | `fradapt.py` | CPU | ~1 min |
| 16 | cef | bge-reranker-base continued from `ceb` on `fra` + US/India replay | `ce.py` | GPU | ~50 min (Blackwell) |
| 17 | cer2 | cross-encoder 5 (bge-reranker-v2-m3, 1.6M pairs × 2 epochs per fold) | `ce.py` | GPU | ~4 h 30 (Blackwell) |
| 18 | ceq | cross-encoder 6 (Qwen3-Reranker-4B, LoRA r=16, 1.2M pairs × 2 epochs per fold) | `ce.py` | GPU | ~30 h (one Blackwell) |
| 19 | fin5 | stage 2 with `ceb` and with `cef` (both + `cer2` + `ceq`), logit blend tuned on OOF | `train.py`, `blend.py`, `decide.py` | CPU | ~1 h 10 |
| 20 | ceqf | `ceq` continued on `fra2` + US/India replay; re-scores only the unseen-country test pairs | `ce.py` | GPU | ~3 h (Blackwell) |
| 21 | fin6 | the fin5 fold models re-score the test pairs with `ceqf` patched in (no retraining) | `train.py`, `blend.py` | CPU | ~30 min |
| 22 | out | final candidate set = stage-1 p ≥ 0.005 (matching restricted to it); decision rule tuned on `tsim` OOF; unseen-country shift δ = −2; writes both outputs; runs the official validator | `cands.py`, `decide.py` | CPU | ~15 min |

Steps 2–19 all produce out-of-fold predictions on `tsim` (folds by S1 id hash, seed 7), so every feature used by
stage 2 comes from models that never saw that S1's labels.

## 5. Outputs

- Written to `$OUT/matching_results.tsv` and `$OUT/candidate_pairs.tsv`.
- The last command of step 22 checks both files with `student_resource/utils/validate_submission.py`. It must print `PASS`.
- `candidate_pairs.tsv` is the final candidate set: the first-pass candidates (key blocking + sibling expansion + ANN,
  ~103 per S1) filtered by the stage-1 model at p1 ≥ 0.005. That gives **4.47 candidates per S1** on test (7.74M pairs;
  recall ceiling 98.84% on validation). The final matching model only runs on these pairs, and every match is one of them.
- The cut was chosen on validation with `ana/cands.py`: p1 ≥ 0.001 gives 5.30 per S1 (F0.5 +0), 0.005 gives 4.47
  (−0.00004), 0.01 gives 4.21 (−0.0001).
- Validation metrics are saved in `$ER_WORK/dec_final.json`: macro F0.5 overall, per country, F0.4, and the predicted-empty rate.

## 6. Source map (`src/`)

| file | role |
|---|---|
| `metric.py` | macro F-beta exactly as the challenge README |
| `norm.py` | name / address normalization: legal forms, street types, states, Indic transliteration, France rules (departments → regions, N°, ordinals, elisions, hyphens) |
| `dicts.py` | native-script → Latin token dictionary learned from matched training pairs |
| `prep.py` | TSV → parquet, parallel normalization |
| `mksim.py` | test-like validation split from train (drops 19% of S1 so their S2/S3 records become orphans) |
| `block.py`, `ext.py` | compound-key blocking, IDF key score, top-K caps |
| `expand.py`, `kx.py` | sibling expansion: re-blocking around confident matches |
| `ann.py`, `annka.py` | ANN candidate pass and its union with the key candidates |
| `feats.py`, `feats2.py`, `xf.py` | pair features: string similarities, house numbers, legal form, name commonness, hard-negative signals |
| `train.py` | LightGBM with 5-fold GroupKFold by S1; OOF + fold-averaged test scores; `sub=` cascade mode |
| `cands.py` | final candidate set: stage-1 learned filter (p1 ≥ thr); restricts final predictions to it |
| `ctx.py` | competition features from stage-1 scores (rank, margin to runner-up, candidate counts) |
| `sup.py` | sibling-support features (agreement with an S1's other confident matches) |
| `ce.py` | cross-encoder fine-tuning / scoring of the uncertain band, 2-fold by S1; `lora=` (4B reranker), `init=` (continue a saved model), `extra=` (self-training pairs), `unseen=1` (re-score only unseen-country test pairs) |
| `fradapt.py` | French self-training set (teacher pseudo-labels, exact S1–S1 negatives, noised S1 positives) |
| `blend.py`, `tune.py` | logit blend of stage-2 variants (weight tuned on OOF); Optuna for stage-2 parameters |
| `decide.py` | one-to-one assignment with margin, expected-F0.5 set selection with an empty option, writes outputs |
| `gate.py`, `xf2.py`, `llm.py` | experiments not used by the final pipeline (entity gate, extra features, LLM judge) |
| `snap.py` | version snapshot (code + metrics) for the experiment log |
