# Business Entity Resolution: end-to-end pipeline

This folder regenerates both submission files from the challenge data:

- `matching_results.tsv`: one row per test Source-1 (S1) entity with its matched Source-2/3 ids (may be empty).
- `candidate_pairs.tsv`: the candidate set the final matching model scores, 9,187,520 pairs (5.30 per S1).
  Every match is one of these candidates.

| | |
|---|---|
| Submitted version | `tb5_frm2`: team blend of three stage-2 members, France calibration shift δ = −2 |
| Validation macro F0.5 (test-like split of train, US + India) | 0.991528 (US 0.991122, India 0.992135) |
| Public leaderboard macro F0.5 | 0.98834 |
| Test links written | 5,828,829 for 1,732,544 S1 entities |

```
data ─► normalization ─► candidate generation ─────────────► stage-1 LightGBM ─► cascade filter p1 ≥ 0.001
        (rules + learned   (compound blocking keys, sibling     (pair features)      = candidate_pairs.tsv
         transliteration)   expansion, multilingual ANN)                                     │
                                                                                            ▼
matching_results.tsv ◄─ decision rule ◄─ team blend ◄─ stage-2 members on the same pairs:
                        (one-to-one,     (logit avg)    fin5s (ours: LightGBM ×2 + 6 fine-tuned cross-encoders)
                         expected F0.5)                 cxQ, fx (teammates' LightGBMs, scores imported in step 19)
```

The final scores are a logit average of three stage-2 members that score exactly the same candidate pairs. This folder
contains all code for the shared pipeline, our member `fin5s`, the team blend and the decision rule. The two teammate
members enter as score files (Section 5, steps 19–20).

The methodology is described in `Documentation_template.md` at the root of the submission.

## 1. Requirements

- **OS and Python:** Linux x86_64, Python 3.11.
- **GPU:** one NVIDIA GPU with a CUDA 12.8 driver.
  - Steps 3–15 fit on a 24 GB GPU (we used A30, A6000 and RTX PRO 6000 cards).
  - Step 16 (4B reranker, LoRA, bf16, batch 32) ran on 97 GB RTX PRO 6000 Blackwell GPUs; use a GPU of that class.
- **CPU and RAM:** 32 cores and about 230 GB RAM for the CPU steps.
- **Disk:** about 200 GB for intermediate files, plus 12 GB for the pretrained models.
- **Network:** only needed for installing packages and downloading the models. The pipeline itself runs offline.

## 2. Setup

```bash
cd code/business_entity_resolution
python3.11 -m venv er-env
er-env/bin/pip install -r requirements.txt              # pinned versions; torch 2.7.1 + CUDA 12.8
er-env/bin/python src/download_models.py out=models     # the only step that downloads anything
```

`download_models.py` fetches each model at the exact commit we used. All are general-purpose models, fine-tuned by
the pipeline only on the provided data:

| model | licence | parameters | commit | used for |
|---|---|---|---|---|
| `intfloat/multilingual-e5-small` | MIT | 118M | `614241f6` | ANN candidate pass (frozen, never fine-tuned) |
| `cross-encoder/ms-marco-MiniLM-L12-v2` | Apache-2.0 | 33M | `7b023523` | cross-encoder `cev` |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Apache-2.0 | 118M | `1427fd65` | cross-encoder `cem` |
| `BAAI/bge-reranker-base` | MIT | 278M | `2cfc18c9` | cross-encoders `ceb`, `cef` |
| `BAAI/bge-reranker-v2-m3` | Apache-2.0 | 568M | `953dc6f6` | cross-encoders `cer` (teacher input only), `cer2` |
| `Qwen/Qwen3-Reranker-4B` | Apache-2.0 | 4.0B | `22e68366` | cross-encoder `ceq` (LoRA) |

Every model is MIT or Apache-2.0 licensed and at most 4B parameters (the limit is 8B). No hosted API is called.

## 3. Data

Use the challenge dataset unchanged:

```
<dataset>/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
<dataset>/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
```

No other data is read.

## 4. Run end-to-end

```bash
cd code/business_entity_resolution
ER_DATA=/path/to/student_resource/dataset PY=er-env/bin/python bash src/run_all.sh
```

The script runs 21 steps in order and prints every command. It stops at the first error.

| variable | meaning | default |
|---|---|---|
| `ER_DATA` | dataset directory containing `train/` and `test/` (required) | |
| `ER_WORK` | intermediate files | `./work` |
| `OUT` | where the two output files are written | `./output` |
| `ER_PRETRAINED` | pretrained models written by `download_models.py` | `./models` |
| `PY` | Python interpreter of the environment above | `python` |
| `VALIDATOR` | official `validate_submission.py` | `$ER_DATA/../utils/validate_submission.py` |
| `FROM` | resume at this step; earlier steps are skipped and their files in `ER_WORK` reused | `1` |
| `TEAM_SCORES` | the teammates' stage-2 score files for step 19 (`oof_kcxQ`, `pte_kcxQ`, `oof_kfx`, `pte_kfx`, `.parquet`) | `./team_scores` |

When the run finishes:

- `$OUT/matching_results.tsv` and `$OUT/candidate_pairs.tsv` are written.
- The official validator runs on both files and must print `PASS`.
- `$ER_WORK/dec_tb5.json` holds the validation scores: macro F0.5 overall and per country, F0.4, and the predicted-empty rate.

The submitted files are in `../../output/`. A rerun reproduces them up to the small differences caused by
non-deterministic GPU training (Section 7).

**Runtime:** about 2.5 days on one machine with one large GPU. Most of it is step 16 (about 35 GPU-hours).

**Two GPUs:** each cross-encoder trains two independent halves. On two GPUs you can run the halves in parallel,
which is how we ran them:

1. Run the step's `ce.py` command twice at the same time, adding `half=0` to one and `half=1` to the other (one GPU each).
2. Then run `python src/ce.py merge=1 name=<name> tag=<tag>`.
3. Continue with `FROM=<next step>`.

This is equivalent to running both halves in one process.

## 5. Steps

`tsim` is the test-like validation split built from train in step 1. Every score passed from one stage to the next is
out-of-fold on `tsim` (folds by S1 id), so no stacked feature has seen that S1's labels.

| # | step | what it does | scripts | resources | time |
|---|---|---|---|---|---|
| 1 | base | TSV → parquet; transliteration dictionary learned from train matches; normalization; test-like split `tsim` (19% of train S1 dropped so their records become orphans, as in test); compound-key blocking, top-100 per S1 / top-10 per S2/S3; pair features | `prep`, `dicts`, `norm`, `mksim`, `block`, `ext`, `feats`, `xf`, `feats2` | CPU | 1 h 10 |
| 2 | a1 | stage-1 LightGBM on the key candidates | `train` | CPU | 1 h |
| 3 | cev | cross-encoder on a1's uncertain band (0.003 < p < 0.997), normalized text | `ce` | GPU | 1 h 15 |
| 4 | kx | sibling expansion: re-blocking of unassigned records against S1 profiles enriched with their confident matches; features of the new pairs | `expand`, `kx`, `feats`, `xf`, `feats2` | CPU | 15 min |
| 5 | ann | frozen multilingual-e5-small embeddings; top-3 S1 per S2/S3 record within its country | `ann` | GPU | 1 h 10 |
| 6 | b1 | stage-1 LightGBM on key + expansion candidates | `train` | CPU | 1 h 10 |
| 7 | ka | ANN pairs: features, own stage-1 model `b1ann`, union `b1ka`; competition and sibling-support features | `annka`, `feats*`, `xf`, `train`, `ctx`, `sup` | CPU | 25 min |
| 8 | cem | cross-encoder, multilingual MiniLM, raw text | `ce` | GPU | 1 h 25 |
| 9 | ceb | cross-encoder, bge-reranker-base, raw text | `ce` | GPU | 3 h 40 |
| 10 | opt | Optuna, 30 trials of stage-2 LightGBM parameters (validation logloss) | `tune` | CPU | 20 min |
| 11 | cer | cross-encoder, bge-reranker-v2-m3 (600k pairs); input of the teacher only | `ce` | GPU | 4 h 15 |
| 12 | fin2 | teacher stage-2 model for the self-training set | `train`, `decide` | CPU | 35 min |
| 13 | fra | self-training set for the country absent from train (see below) | `fradapt` | CPU | 1 min |
| 14 | cef | bge-reranker-base continued from `ceb` on the self-training set plus US/India replay | `ce` | GPU | 50 min |
| 15 | cer2 | cross-encoder, bge-reranker-v2-m3 (up to 3M pairs, 2 epochs) | `ce` | GPU | 4 h 30 |
| 16 | ceq | cross-encoder, Qwen3-Reranker-4B with LoRA r=16, bf16 | `ce` | GPU (97 GB) | 35 h |
| 17 | fin5 | stage-2 LightGBM with `ceb` (fin5a) and with `cef` (fin5b), both with `cev`, `cem`, `cer2`, `ceq`; logit blend weight and decision rule tuned on validation | `train`, `blend`, `decide` | CPU | 2 h |
| 18 | fin5s | our team member: stage 2 `fin3` (ceb + cer2, no ceq) → sibling support `sup3` recomputed from its scores → `fin5sa` (fin5a with `sup3`) → logit blend 0.8 fin5sa + 0.2 fin5b | `train`, `sup`, `blend` | CPU | 2 h 30 |
| 19 | team | copy the teammates' stage-2 scores (`kcxQ`, `kfx`) from `TEAM_SCORES`; gate: exact row counts, 100% pair-key overlap with `fin5s`, identical labels | `kcheck` | CPU | 2 min |
| 20 | tb5 | team blend in logit space: US/India weights by simplex search on validation F0.5 (result 0.7 cxQ / 0.3 fin5s / 0 fx); France equal thirds | `team_blend` | CPU | 30 min |
| 21 | out | decision rule on the blended test scores, with calibration shift δ = −2 for countries absent from train; writes both files; runs the validator | `decide` | CPU | 15 min |

Times are measured on our cluster for one GPU and 32 CPU cores. Steps 3, 9 and 11 ran on an A30, step 8 on an A6000,
and steps 14–16 on an RTX PRO 6000 Blackwell; for those three, the two halves' times are added.

**Candidate set:**
- The final matching model scores exactly the pairs that stage 1 (steps 2–7) keeps at p1 ≥ 0.001: 9,187,520 test pairs.
- Step 21 writes these same pairs to `candidate_pairs.tsv` (`decide.py apply`), so every match is a candidate by construction.

**Self-training set (step 13):** France has no training labels, so this set is built only from the provided records:
- the teacher's confident predictions on French test pairs;
- exact negatives: two different S1 records that share an address or name;
- exact positives: an S1 record against a noised copy of itself.

The organizers' Q&A allows self-training on the test records and synthetic pairs built from the provided data.

**Teammate members (step 19):**
- `cxQ` (teammate A) and `fx` (teammate B) are LightGBM stage-2 models trained by our teammates on the same tsim folds
  and the same 9,187,520 test pairs, with their own cross-encoders (bge-reranker-v2-m3 variants, plus our `ceq`
  scores in cxQ). Their inputs are described in `Documentation_template.md`.
- Their training code is kept in the teammates' own repositories and is not part of this folder. Step 19 reads
  their score files: `oof_<tag>.parquet` (tsim out-of-fold, columns id1, id2, p) and `pte_<tag>.parquet`
  (test, columns id1, id2, p), with ids encoded by `block.enc`.
- With these four files, steps 19–21 rebuild the submitted `matching_results.tsv` and `candidate_pairs.tsv`
  byte for byte from our `fin5s` scores (checked on our cluster).

## 6. Source files (`src/`)

| file | role |
|---|---|
| `run_all.sh` | entry point: the 21 steps above |
| `download_models.py` | downloads the six pretrained models at pinned commits |
| `prep.py` | TSV → parquet; parallel normalization |
| `norm.py` | name/address normalization: legal forms, street types, US/Indian states, Indic transliteration, French rules (department → region, N°, ordinals, elisions, hyphens) |
| `dicts.py` | native-script → Latin token dictionary learned from matched training pairs |
| `mksim.py` | test-like validation split |
| `block.py`, `ext.py` | compound-key blocking, IDF-weighted key score, top-K caps, recall report |
| `expand.py`, `kx.py` | sibling expansion |
| `ann.py`, `annka.py` | ANN candidate pass and its union with the key candidates |
| `feats.py`, `xf.py`, `feats2.py` | pair features: string similarities, house numbers, legal form, name commonness, hard-negative signals |
| `train.py` | LightGBM, 5-fold GroupKFold by S1; OOF and fold-averaged test scores; cascade option `sub=` |
| `ctx.py` | competition features from stage-1 scores (rank, margin to the runner-up, counts) |
| `sup.py` | sibling-support features (agreement with the S1's other confident matches) |
| `ce.py` | cross-encoder fine-tuning and scoring, 2-fold by S1 (`lora=`, `init=`, `extra=`, `half=`, `merge=1`) |
| `fradapt.py` | self-training set for countries absent from train |
| `tune.py` | Optuna search for stage-2 parameters |
| `blend.py` | logit blend of stage-2 models, weight tuned on validation F0.5 |
| `kcheck.py` | gate for an imported stage-2 score file (row counts, key overlap, labels, correlations) |
| `team_blend.py` | team blend: simplex weight search on validation for seen countries, fixed weights for unseen ones |
| `decide.py` | one-to-one assignment, expected-F0.5 set selection with an empty option, calibration shift, writes both files |
| `metric.py` | macro F-beta exactly as the challenge defines it (`python src/metric.py` checks the README example, 0.714) |

## 7. Reproducibility notes

- **Seeds are fixed:** fold assignment, validation split, LightGBM, Optuna, and samples.
- **CPU steps** give the same results on the same inputs and CPU count.
- **GPU steps are not bit-deterministic.** Cross-encoder training shuffles batches and uses mixed precision, so a rerun gives
  equivalent but not byte-identical scores, and the final files can differ in a small number of rows.
- **How the submitted files were produced:** the same commands ran as SLURM jobs, with cross-encoder halves on two GPUs in
  parallel as described in Section 4.
- **Versions:** every package matches `requirements.txt`, with one exception. Steps 3, 5, 8 and 9 ran with torch 2.5.1+cu121,
  and the later GPU steps with torch 2.7.1+cu128, which is pinned here and runs every step.
- **Data use:** no external data, API or lookup is used.
  - The dictionaries in `norm.py` are small and hand-written: legal forms, street types, US/Indian state codes,
    French department → region names, and eight alternate spellings of Indian city names (Bombay → Mumbai).
    There is no city → region or postal-code table.
  - The transliteration dictionary is learned from the training matches.
  - Test records are used without labels: blocking statistics, and self-training as allowed by the organizers.
