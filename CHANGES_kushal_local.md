# Technical changes on branch `kushal/local` vs the cloned repo

Amazon ML Challenge 2026 — Business Entity Resolution. Code changes, architecture changes,
methodology and strategy. Written for an engineer picking this branch up cold.

---

## 1. Portability refactor (cluster → any machine)

The cloned code embedded one specific SLURM cluster: `/scratch/m25csa019`, `/csehome/...`,
`SLURM_CPUS_PER_TASK`, and `sbatch`/`sacct` job control. Every one of those is gone.
**No pipeline logic was altered** — the refactor is path/scheduling only.

### `er/src/paths.py` — new module
Single source of truth for all filesystem locations and the CPU count, resolved from environment
variables with sensible defaults:

```
ER_ROOT   scratch root for everything large      (default /data/$USER/amazon_er)
ER_WORK   intermediates                          ER_DATA  dataset dir
ER_OUT    submissions                            ER_VERS  version snapshots
ER_MODELS fine-tuned weights                     ER_CPUS  workers / LightGBM threads
```

Every module in `src/` and `ana/` now does `from paths import W` instead of holding a literal.
`ana/*.py` additionally had their `sys.path` bootstrap fixed (several pointed at `ana/` rather than
`src/`, and most were missing `import os` once the path became computed).

### `er/jobs/dag.py` — rewritten: local DAG runner replacing SLURM
Same spec-file concept, new execution model. Format gains a resource column:

```
name | deps | <cpu|gpu> <mem_gb> | cmd1; cmd2; ...
```

Scheduling behaviour:
- **CPU stages run one at a time**, using all cores (they are already internally parallel).
- **GPU stages run concurrently, one per GPU**, via `CUDA_VISIBLE_DEVICES`, each pinned to
  `ER_CPUS=8` for its dataloader.
- **RAM admission control**: a stage starts only while the sum of declared `mem_gb` over running
  stages stays under `ER_MEM_GB` — this box has 125 GB and is shared, versus the cluster's 240 GB.
- **Resumable**: a completed stage writes `$ER_LOGS/<stage>.done` and is skipped on re-run;
  `only=a,b` runs a subset. Failures propagate to dependents; independent branches continue.

Verified against a synthetic DAG covering dependency order, GPU assignment, failure propagation
and resume.

### `er/jobs/final.dag` — converted, and stages split
Same stages, same arguments as the teammate's SLURM version. The monolithic `base` stage (14
chained commands, several hours) was split into **`prep` / `blk` / `base`** so a failure late in
the chain no longer discards everything before it.

### Other rewrites
- `er/jobs/env.sh` — the only file to edit per machine; exports the variables above.
- `er/run_all.sh`, `er/jobs/release.sh` — local equivalents of the SLURM wrappers.
- `er/requirements.txt` — merged the teammate's pins, added `xgboost`. **torch is installed
  separately from the cu128 index**: their pinned `torch 2.5.1+cu121` cannot execute on Blackwell
  (sm_120) GPUs at all.
- `er/README.md` — merged their model-licence and step tables with the new environment/runner docs.

### Deleted (cluster-only; still present on the teammate's branch)
`jobs/*.sbatch`, `chain.sh`, `queue.sh`, `after_kas1.sh`, `after_fin.sh`, `chain_ann*.txt`,
`q_phase*.txt`, `resume*.dag`, `ana/disk.py`.

### Environment gotchas worth keeping
- **CUDA orders GPUs by speed, not by `nvidia-smi` index.** Here `CUDA_VISIBLE_DEVICES=0` is the
  96 GB card and `=1` is the 48 GB card — the reverse of `nvidia-smi`. Assuming otherwise caused an
  OOM on the first launch.
- A watchdog terminates pipeline jobs if free RAM drops below 10 GB, since the machine is shared.

---

## 2. `er/src/ce.py` — cross-encoder training/scoring extensions

All additions are backwards compatible; existing invocations behave identically.

| option | purpose |
|---|---|
| `half=0\|1` | Run a single OOF fold-half. The two halves are independent models, so they now train **concurrently, one per GPU**, roughly halving wall-clock. `ce.py merge tag=… name=…` recombines them (OOF concatenated, test predictions averaged). |
| `amp=bf16` | bf16 autocast instead of fp16. Large models overflow fp16 more readily; `GradScaler` is auto-disabled under bf16 since it is unnecessary. |
| `sep=` | **Decoder-model support.** HF tokenizers concatenate a text pair with *no separator* for causal LMs, so two records would run together into one string. `sep=` joins them explicitly as a single sequence. |
| `ml=` | Max token length (was hard-coded 128). |
| — | **pad-token fix**: decoder checkpoints ship without `config.pad_token_id`, which breaks last-token pooling in batched sequence classification. Now populated from the tokenizer when absent. |

The last three exist because decoder-based rerankers (Qwen3) are a different architecture class
from the XLM-R encoders the original code assumed. Compatibility was verified before spending GPU
time — `Qwen3-Reranker-0.6B` loads as `Qwen3ForSequenceClassification` with a fresh scoring head.

---

## 3. Architecture change: a larger multilingual reranker in stage 2

**Change:** add `BAAI/bge-reranker-v2-m3` (568M, Apache-2.0, XLM-R-large) as an additional
stage-2 feature column alongside the existing `cev` / `cem` / `ceb`.

**Rationale (evidence-based, not speculative):** every gain in this project's history came from
neural language capacity — ANN retrieval +0.00287, `cem` +0.00072, `ceb` gave France +0.009 — while
Optuna, the entity gate and XGBoost blending each returned ~0.000. The incumbent `ceb`
(bge-reranker-*base*, 278M) scores French pairs at median −8.13 versus −3.06 on training data,
i.e. it is systematically unconfident on French text, and France is where the remaining loss is.

**Training configuration and why:**
- Band 0.003–0.997 on stage-1 score, `ntr=3M`, 2 epochs — matched to the teammate's `ceb` run so
  the comparison isolates the model change.
- `txt=raw` — native scripts, casing and punctuation preserved; multilingual models need the
  original text, not the ASCII-folded normalization.
- bf16; batch 256 on the 96 GB GPU and 112 on the 48 GB GPU, with learning rate scaled accordingly
  (1.5e-5 / 1.1e-5).
- 2-fold by S1 hash, so the feature is out-of-fold and cannot leak into stage 2.

**Measured outcome (tag `cx1`, submitted 2026-09-26):**

| | validation macro F0.5 (US+India) | leaderboard |
|---|---|---|
| `fin` (previous best) | 0.99089 | 0.986946 |
| `cx1` = `fin` + `cx` column + `fr=-2` | **0.99106** | **0.987449** |

Band AUC 0.9835 / 0.9832 per fold-half — the best recorded here. Stage-2 gain share: **`cx` 0.0526**
while `ceb` fell to 0.0064, `cem` 0.0038, `cev` 0.0016. The new model **replaced** the older
cross-encoders rather than complementing them, exactly as the redundancy analysis predicted
(corr 0.984 with `ceb`; residual after regressing out `ceb`+`cem` has AUC 0.46, i.e. no independent
signal). The remaining upside is therefore an *architecturally different* reranker, not a bigger
one of the same family.

**Also prepared:** `Qwen/Qwen3-Reranker-0.6B` (596M, Apache-2.0) as an architecturally *diverse*
second model — decoder rather than encoder, different tokenizer and pretraining corpus. Diversity
is what made the teammate's three-CE stack work.

**Compliance checked via the HF API:** all candidate weights MIT/Apache-2.0. **`Qwen3-Reranker-8B`
is 8,189M parameters — over the challenge's 8B cap — and was excluded deliberately.**

---

## 4. New feature engineering: `er/src/xf3.py` (built, measured, **not adopted**)

"Discriminative-difference" features: is what *differs* between two names rare or generic?

Motivated by the documented France failure mode — distractors *substitute* one generic word
("Nantes College" vs "Nantes Amicale" = different businesses) whereas true matches *drop* words
("Green Creative Dynamics" vs "Green Creative" = same business). Aggregate similarity features
(`co_widf`, `co_jac`, `co_cov12/21`) cannot separate these, since both look like "most tokens
shared, one differs".

Emits `d_n1`, `d_n2`, `d_both` (substitution vs truncation), `d_imax1/2`, `d_imax_inter`, and
`d_gap` — the IDF of the shared tokens minus the IDF of the differing ones. **Token IDF is computed
per country**, so "ecole" is correctly generic inside France despite never appearing in US/India.

**Outcome: +0.00002 OOF; features rank at the bottom of the importance list.** The signal is real
(substitution pairs match 66% vs 78%) but stage-1 probability already tracks it almost exactly
within every bucket — the model had learned it. Retained for the experiment log only.

---

## 5. Methodology changes

### Fast-iteration loop (the biggest practical change)
The teammate's `s2mat_{tsim,test}.parquet` are the complete stage-2 feature matrices (8.19M / 9.19M
rows × 117 cols). Consequence: **testing a new feature costs one extra column plus a ~22-minute
stage-2 retrain, not a 12-hour end-to-end pipeline run.** All experimentation now runs through this
loop.

To make it self-contained, `oof_b1ka` / `pte_b1ka` were **reconstructed locally from `s2mat`'s
`c_p` column** (which is the stage-1 score). This lets `ce.py`'s existing band-selection and
2-fold-by-S1 OOF logic run without needing the teammate's stage-1 artifacts.

### Baseline-parity gate
Before trusting any measurement, a local stage-2 retrain was required to reproduce the reference
run exactly. It does — 0.99089 / US 0.99048 / India 0.99152, identical to five decimals. Every
subsequent delta is therefore attributable to the change under test rather than to environment
drift.

### Label-free calibration diagnostics (new technique)
The model's probabilities are near-exactly calibrated on US/India, which turns two quantities into
unbiased estimators that need **no labels**:
- `mean(Σp per S2/S3 record)` predicts the true owner rate (0.8723 vs 0.8724 actual)
- `mean(p | selected)` predicts true precision (0.9987 vs 0.9987 actual)

Applied to France — where no labels exist — these yield a *self-assessment* that can be compared
against the leaderboard-implied score, quantifying the model's overconfidence on unseen data.
A companion structural check counts violations of the one-to-one constraint (Σp > 1 for a record,
logically impossible), which is 5–6× more frequent on France.

### Leave-one-country-out protocol
Train on one country, score the other, versus an equal-data in-country control. Isolates the
"unseen country" penalty from the "less data" penalty. Used to test whether France's gap is a
transfer problem.

### France-like validation proxy
`tsim` contains no France, so France changes are unmeasurable offline. Constructed a proxy subset
of US/India entities matching France's *ambiguity profile* (≥7 candidates, ≤55% confident scores)
to at least test decision-rule and feature changes against France-shaped difficulty.

### Validation→leaderboard amplification (now measured, not assumed)
`tsim` contains no France, so a validation delta systematically **understates** the leaderboard
delta for anything that improves multilingual text handling. With `cx1` this is now quantified:

| | validation Δ | leaderboard Δ | ratio |
|---|---|---|---|
| `ceb` (teammate) | +0.00014 | +0.0015 | ~10× |
| `cx1` | +0.00016 | **+0.000503** | **~3.1×** |

Backing the countries out of `cx1`: US+India contributed ~0.00014, so **France gained ≈ +0.0025**
(≈0.964 → ≈0.966).

**Practical rule: multiply a validation gain by ~3 to estimate the leaderboard gain**, and do not
apply a flat +0.0005 adoption bar — it would have rejected both `ceb` and `cx1`. An a-priori bar of
+0.0002 was set before `cx1` and proved too strict (`cx1` measured +0.00016 and was worth
+0.000503); it was overridden deliberately.

---

## 6. Strategy change: `candidate_pairs.tsv` reinterpretation

Amazon's 2026-09-26 update made `candidate_pairs.tsv` count toward the **final ranking**, with
smaller candidate sets ranked higher.

The pipeline is a cascade: blocking → **stage-1 ranker filters at p ≥ 0.001** → stage-2 matcher
produces the results. The spec defines candidate_pairs as *"the exact set of records you feed into
your matching model for inference… not the raw output of an early blocking pass you later filter
further. If your pipeline has several blocking/filtering stages, candidate_pairs.tsv is the last
one."*

| set | pairs per S1 |
|---|---|
| raw blocking output (what was being submitted) | 103.28 |
| cascade set actually scored by stage 2 | **5.30** |

A **19.5× reduction**, validator PASS with no subset warnings, and no change to
`matching_results.tsv` — so it costs no submission. The methodology document should state **both**
numbers so the reduction is transparent rather than something an auditor must infer.

---

## 7. Where the evidence says effort should go

Measured and **ruled out** — these need not be retried:

| hypothesis | measurement |
|---|---|
| Decision rule / threshold wrong for France | margin×m grid on France-like entities: +0.00005 |
| Model cannot generalize to an unseen country | LOCO penalty only 0.0005 |
| True matches missing from the candidate set | 100% record coverage in all three countries |
| Name-difference IDF features | +0.00002 |
| Model under-fit in France's "contested" regime | **refuted** — calibration holds there (±0.02), and contested pairs are 24% of training, not a rare corner |

**Still open and quantified:** a perfect matcher restricted to the candidates already scored would
reach **0.99666** versus the current **0.99089** — **0.0058 of headroom on US+India**, which is
larger than the gap to rank 1. Since `tsim` covers US+India, that headroom is *measurable offline*,
unlike France. This reframes the effort: model quality, not France-specific post-processing.

**Constraint held throughout:** training data only — no pseudo-labels and no training on test
inputs (standing team decision), and all weights MIT/Apache-2.0 under 8B parameters.

---

## 8. Outcome and what remains

**Shipped 2026-09-26:** `cx1` — leaderboard **0.987449** (from 0.986946), the team's best.
Gap to rank 1 (0.990788) is now **0.00334**; at the measured ~3× amplification that needs roughly
**+0.0011 more on validation**, which sits well inside the **0.0058** of US+India headroom (§7).

**One small addition made during the run:** `decide.py apply` requires `dec_<tag>.json`, which only
`decide.py tune` writes. Since the decision grid is verified saturated (§7), `dec_cx1.json` was
created by **reusing `fin`'s tuned config** rather than spending ~20 min re-tuning. Any future tag
can do the same; re-tune only if the decision rule itself changes.

**Next, in order:**
1. `Qwen/Qwen3-Reranker-0.6B` — decoder architecture, different tokenizer and pretraining corpus.
   Chosen specifically because `cx` proved redundant with `ceb`; diversity, not size, is what is
   still unexploited. The `ce.py` decoder support (§2) exists for this.
2. Blend the two rerankers via `src/blend.py` (weights tuned on OOF).
3. Re-tune the France shift **for the new model** — `fr=-2` was fitted to `fin`, and a changed
   score distribution moves the optimum.
4. Seed-average stage-2 (~+0.0002 validated, cheap).

**Constraint unchanged:** training data only; all weights MIT/Apache-2.0 under 8B parameters.
