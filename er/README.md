# Business Entity Resolution — reproducible pipeline

Links every Source-1 business to its Source-2 / Source-3 records using only name, address and country, and writes
the two submission files:

- `matching_results.tsv`: one row per Source-1 entity, its matched S2/S3 ids (may be empty)
- `candidate_pairs.tsv`: the blocking candidate set the matching model scores

**Pipeline in one line:** normalization → multi-pass blocking (compound keys, then sibling expansion, then ANN) →
pair features → stage-1 LightGBM → three cross-encoders on the uncertain band → cascade stage-2 LightGBM →
one-to-one assignment with an expected-F0.5 set rule tuned on out-of-fold predictions.

No external data, APIs or lookups are used. Dictionaries are learned from the training pairs, and France rules are
hand-written. Test inputs are only used as inference inputs: no labels exist for them and nothing is trained on them.

## 1. Environment

```
conda create -p <env> python=3.11
<env>/bin/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
<env>/bin/pip install -r requirements.txt
```

**Pretrained weights**
- Download these once into `$HF_HOME`. GPU jobs run with `HF_HUB_OFFLINE=1`, so they must already be cached.
- Every model is used as published, with no business data added.

| model | licence | used in |
|---|---|---|
| `intfloat/multilingual-e5-small` | MIT | ANN pass (frozen, never fine-tuned) |
| `cross-encoder/ms-marco-MiniLM-L12-v2` | Apache-2.0 | cross-encoder `cev` |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Apache-2.0 | cross-encoder `cem` |
| `BAAI/bge-reranker-base` | MIT | cross-encoder `ceb` |

```
HF_HOME=<cache> <env>/bin/python -c "from huggingface_hub import snapshot_download as d; [d(m) for m in ['intfloat/multilingual-e5-small','cross-encoder/ms-marco-MiniLM-L12-v2','cross-encoder/mmarco-mMiniLMv2-L12-H384-v1','BAAI/bge-reranker-base']]"
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
- The PyTorch build above supports sm_80/sm_86 GPUs such as A30, A6000 and A100. It does **not** support Blackwell (sm_120).

```
ER_DATA=<dataset> ER_WORK=<scratch> HF_HOME=<cache> PY=<env>/bin/python OUT=<outdir> bash run_all.sh
```

**Option B, SLURM (how the submitted files were produced):**
- `jobs/final.dag` lists the 11 jobs and their dependencies.
- `jobs/dag.py` submits each job when its inputs are ready, respecting the per-user job limit.
- CPU jobs use `jobs/multi.sbatch`; GPU jobs use `jobs/gpu.sbatch`.

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
| 11 | fin | cascade stage 2: all pairs with stage-1 p ≥ 0.001 and all features, 5-fold; decision rule tuned on `tsim` OOF; writes both outputs; runs the official validator | `train.py`, `decide.py` | CPU | ~1 h 30 |

Steps 2–10 all produce out-of-fold predictions on `tsim` (folds by S1 id hash, seed 7), so every feature used by
stage 2 comes from models that never saw that S1's labels.

## 5. Outputs

- Written to `$OUT/matching_results.tsv` and `$OUT/candidate_pairs.tsv`.
- The last command of step 11 checks both files with `student_resource/utils/validate_submission.py`. It must print `PASS`.
- `candidate_pairs.tsv` is exactly the scored `ka` set: key blocking + sibling expansion + ANN.
- Validation metrics are saved in `$ER_WORK/dec_fin.json`: macro F0.5 overall, per country, F0.4, and the predicted-empty rate.

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
| `ctx.py` | competition features from stage-1 scores (rank, margin to runner-up, candidate counts) |
| `sup.py` | sibling-support features (agreement with an S1's other confident matches) |
| `ce.py` | cross-encoder fine-tuning / scoring of the uncertain band, 2-fold by S1 |
| `decide.py` | one-to-one assignment with margin, expected-F0.5 set selection with an empty option, writes outputs |
| `blend.py`, `gate.py`, `tune.py`, `xf2.py` | experiments not used by the final pipeline (logit blending, entity gate, Optuna, extra features) |
| `snap.py` | version snapshot (code + metrics) for the experiment log |
