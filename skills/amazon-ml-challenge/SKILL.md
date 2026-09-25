---
name: amazon-ml-challenge
description: Competition co-pilot for the Amazon ML Challenge and similar 72-hour ML hackathons (Kaggle-style tabular, text, image or multimodal problems scored on a leaderboard). Drives the path from problem statement to a top-ranked submission - metric-aware loss and post-processing, EDA, leak-free folds, feature engineering from catalog text, kNN/target-encoding features, GBDT training with Optuna hyperparameter tuning, multimodal fine-tuning, blending, threshold/multiplier calibration, submission checks and the approach write-up - using tested scripts plus the documented tricks of past top-ranked teams (2021 to 2025). Use this skill whenever the user mentions the Amazon ML Challenge, an ML hackathon or competition, a leaderboard, SMAPE/MAPE/F1 optimisation, product price or attribute prediction, hyperparameter tuning for a competition, improving an OOF or leaderboard score, SageMaker XGBoost/Autopilot for a challenge, or pastes a competition problem statement or dataset, even if they don't name the skill.
---

# Amazon ML Challenge co-pilot

Goal: get from an unseen problem statement to a top-3-calibre solution as fast as possible. Top teams in
past editions did not win with secret models. They won on metric geometry, clean data, honest validation,
diverse ensembles, strict post-processing and a compute plan that finished test inference. This skill
packages those levers as a workflow plus scripts that already work, so hours go into experiments, not
boilerplate.

Scripts live in `scripts/` (run them from the data directory with `python <skill>/scripts/x.py key=value`;
they share `kv.py` and `metrics.py`, so keep the folder together). All take `key=value` arguments and
print usage in their docstring.

## Step 0: the challenge card (first 30 minutes)

Before any modelling, extract these from the problem statement, rules and sample files. Write them at
the top of the working notes and repeat them into every new session, because every later decision
depends on them:

- **Metric**: exact formula, direction (min/max), how empty or invalid predictions are scored.
- **Output format**: columns, id name, units/decimals, sample submission.
- **Data**: rows, columns, modalities (text / images / tabular), target distribution.
- **Rules**: external data policy, model licence and parameter caps (2025: MIT/Apache, at most 8B),
  submission limit, public LB fraction.
- **Compute**: GPUs available and hours; image download size.

Then pick the archetype(s) from `references/archetypes.md` and read `references/metrics.md` for the
metric's loss, blend space and post-processing. If the metric is not in `scripts/metrics.py`, add it
there first so every script scores with the exact official function.

## Step 1: fast baseline pipeline (target: a valid submission in hours 1 to 6)

```bash
S=/path/to/amazon-ml-challenge/scripts
python $S/eda.py train=train.csv test=test.csv target=price id=sample_id        # findings + suggested levers
python $S/make_folds.py train=train.csv target=price id=sample_id group=catalog_content   # ONE shared folds.csv
python $S/download_images.py inp=train.csv col=image_link id=sample_id dir=images        # start early, if images
python $S/parse_catalog.py inp=train.csv col=catalog_content id=sample_id out=f_tr.csv   # and again for test
python $S/knn_feats.py train=train.csv test=test.csv target=price id=sample_id text=catalog_content out=knn
python $S/gbdt_cv.py train=train.csv test=test.csv target=price id=sample_id metric=smape model=cat \
    feats=f_tr.csv:f_te.csv,knn_train.csv:knn_test.csv te=brand drop=catalog_content,image_link tag=cat1
python $S/blend.py train=train.csv test=test.csv target=price id=sample_id metric=smape tags=cat1 out=sub.csv
python $S/sanity.py sub=sub.csv test=test.csv id=sample_id col=price pos=1
```

For classification use `task=clf` in make_folds and a classification metric (`f1`, `acc`, `auc`,
`macro_f1`); blend.py then tunes the decision threshold on OOF automatically. Every script prints OOF
so the user sees a number after every step.

## Step 2: the lever loop (hours 6 to 50)

One lever per experiment, same folds, same metric function. Keep a change only if OOF improves by more
than fold noise (about half the std of fold scores). Log each experiment as
`exp_id | change | OOF | fold std | LB | kept?` because this log becomes the write-up's ablation table.

Pull levers roughly in this order of documented payoff; skip ones the EDA says are irrelevant:

1. **Metric alignment**: log target, L1/Huber (gbdt_cv `tf=log obj=auto`), geometric blending, later the
   exact metric loss. For MAPE-type metrics shrink predictions; for F1 tune thresholds.
2. **Data quality**: review the 200 worst OOF rows; fix systematic label/parse errors; for LoRA/VLM
   fine-tunes build a 1K to 2K hand-curated set (2024 winner: +0.19 F1 from curation alone).
3. **Features from text**: parsed quantities/units/pack (parse_catalog.py; extend UNIT/regex or use an
   LLM extractor when coverage is low), TF-IDF SVD (`tfidf=`), target encoding (`te=`).
4. **Retrieval**: OOF kNN neighbour-target features from text or image embeddings (knn_feats.py).
5. **Hyperparameters**: `trials=30 lr=0.08` to search, then `params=params_<tag>.json lr=0.03` to refit
   (see `references/hyperparams.md` for spaces and the NN/QLoRA recipes).
6. **Representations**: fine-tuned text encoder, then add the image (SigLIP/CLIP), then fusion upgrades;
   `scripts/mm_finetune.py` is the template (Rank 8/20 style). It is an untested template: expect to adapt
   model names and paths, and check licences.
7. **Ensemble**: 3 to 6 diverse members (different families, inputs, targets, losses); blend.py fits
   log-space weights on OOF, prints error correlations, and calibrates multiplier/threshold, clip, snap.

Why this order: early levers are cheap and compound with every later model; fusion and ensembling only
pay off once the data, metric and features are right.

When the user asks "what next?", answer from the EDA signals and the log: name the single lever with the
best expected gain per hour, the exact command or code change, and what OOF movement would confirm it.
`references/archetypes.md` has an EDA-signal → lever table and a synergy matrix of pairs that
reinforce each other.

## Step 3: finish (hours 50 to 72)

Freeze new ideas around hour 50. Retrain finals (seeds averaged), run sharded test inference, blend on
OOF, run sanity.py, upload early, keep a safe backup, and draft the 1 to 2 page approach document from
the experiment log (template in `references/playbook.md`). The final ranking in 2025 also weighed the
document, and finalists are questioned on every design choice.

## Guard rails (each of these has cost teams ranks)

- Trust OOF over the public leaderboard; the public LB was about a third of the test set in 2025.
- Target encoding, kNN features and thresholds must be computed out of fold. The scripts do this;
  hand-written code must too. If OOF suddenly jumps, suspect a leak before celebrating.
- Group folds when near-duplicate listings exist (eda.py reports this).
- Never drop test rows; every id needs a prediction, including rows with missing images or text.
- No external data that the rules forbid (2025 banned external price lookups); verify model licences
  and parameter counts.
- Plan GPU hours for test inference on day 0; image download and VLM inference are the usual bottlenecks.
- Rerun the final pipeline end to end from raw files before the last upload.

## If the organisers provide SageMaker

The intro session used Amazon's SageMaker churn notebook. For the built-in XGBoost container: target
in the first column, no header, all numeric, channels `train`/`validation` in S3; tune with Automatic
Model Tuning; score the test with Batch Transform (no endpoint needed); stop notebooks and delete
endpoints to control cost. Details, code and the scientist's tips mapped to actions:
`references/sagemaker.md`.

## Reference map (read only what the current step needs)

| File | Read when |
|---|---|
| references/metrics.md | Step 0, and whenever choosing a loss, blend space, threshold or post-processing |
| references/archetypes.md | Step 0 (pick the recipe) and when deciding the next lever |
| references/techniques.md | full catalogue of tricks with evidence and measured gains |
| references/hyperparams.md | tuning GBDTs, transformer fine-tunes, QLoRA, SageMaker AMT ranges |
| references/sagemaker.md | SageMaker / Autopilot, or questions about the scientist's session tips |
| references/playbook.md | team timeline, submission discipline, write-up template, pitfalls |
| references/past_solutions.md | what ranked teams did each year, plus learning links |

## House style for this user

Keep new variable names short in code. No emojis. Explanations as bullet points rather than long
paragraphs. Always show the OOF number that justifies a recommendation.
