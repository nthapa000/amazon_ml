# PLAN — Business Entity Resolution (Amazon ML Challenge 2026)

## Status (update after every stage: date | result | OOF F0.5)
Best so far: u4 OOF 0.98517 (tsim) -> submissions/u4. Versions + submissions: VERSIONS.md, submissions/<tag>/matching_results.tsv, /scratch/m25csa019/versions/<tag>/
- [x] 0. Plan stored in repo | 2026-09-25
- [x] 1. Env `/scratch/m25csa019/envs/er` | GPU = NVIDIA A30 24GB | 2026-09-25
- [x] 2. `metric.py` (README example = 0.714) | verified
- [x] 3. `norm.py` + `prep.py` -> normalized parquet (train + test) | 2026-09-25
- [x] 4. `dicts.py` learned transliteration dict (1347 name tokens + 16 addr) | 2026-09-25
- [x] 5. Blocking v2 (b3): recall 0.9728, ceiling 0.9911, 42 cand/S1 | 2026-09-25
- [x] 6. Stage-1 features + LightGBM OOF | 2026-09-25
- [x] 7. Decision (one-to-one + thr 0.7) -> OOF F0.5 0.97803 | 2026-09-25
- [x] 8. First submission file: ~/amazon_ml/submissions/s1/matching_results.tsv (validator PASS) | LB: pending
- [x] 9. Blocking recall: K100 caps (ceiling .9916->.9928) + sibling expansion (->.9936) instead of embedding pass | kx chain running
- [x] 10. Stage-2 competition features | t2 tsim OOF 0.98092 | sub2: ~/amazon_ml/submissions/t2 (validator PASS)
- [x] 11. Expected-F set selection (best: ef, one-to-one margin 0.2); singleton gate: not needed yet
- [ ] 12. LOCO robustness check (France proxy)
- [x] 12b. tsim split (19% S1 dropped) to mimic test orphan rate | 2026-09-25
- [x] 13. Cross-encoder on uncertain band (MiniLM-L6 -> u2 OOF 0.98422, sub3 submissions/u2); L12 wide band (ce12) -> u3 running
- [ ] 14. Pseudo-label self-training on test
- [ ] 15. Optuna + seeds (decision grid already saturated: 0.98519 vs 0.98517)
- [ ] 16. Final end-to-end rerun, validator PASS, zip + Documentation

## Phase 2 status: improvement plan after w2 (full plan: section "Phase 2 plan" below)
- [x] P0. w2a released (OOF 0.98670, sub7); w3a 0.98676 (+0.00006, not adopted); err analysis w2 done; cascade parity exact; noise floor -> from c1a/c1b
- [x] P1. Cascade stage 2: c1a 0.98685 (+0.00015 vs w2a), released; seed noise 0.000004
- [x] P2. Sibling support: c2s 0.98714 (+0.00029)
- [x] P3. Multilingual CE: c2m 0.98757 (+0.00072); combined c2 0.98776 (+0.00091) released
- [x] P4. Recall: NO-GO (expansion round 2 recovers 0.6% of misses; K200 +0.0004 < bar)
- [ ] P5. Optuna + XGBoost blend (+ optional entity gate)
- [ ] P6. Freeze ~18 h before deadline; consolidated run_all.sh end-to-end rerun; package zip
- [ ] P7. (Phase 3) Bigger multilingual cross-encoder BAAI/bge-reranker-base (MIT, 278M), raw text, 2 ep -> c3 = best c2 + ceb  [running, job 41057]
- [x] P8. Error review of c2 -> c4 (prefix-ambiguity + source-aware siblings): 0.98783 (+0.00007, below bar, not adopted)
- [ ] P9. Entity-level gate P(no match) (targets 25% of loss: predicted-empty + singleton FP) + XGBoost member + OOF-weighted blend
- [ ] P10. ANN recall probe: e5-small (MIT) embeddings of name|address, top-20 per country; GO only if >= 15% of the 118k blocked misses recovered
- [ ] P11. Optional, guarded: France adaptation (pseudo-labels from confident test pairs incl. France) and test-side transliteration-dict extension; accept only if tsim OOF does not drop
- [ ] P12. Freeze ~18 h before deadline -> consolidated run_all.sh end-to-end rerun from raw TSVs -> final zip (P6)
Guard rails: fixed folds / tsim split; all stacking inputs OOF; accept a lever only if gain > max(0.0001, 2 x seed noise); LB = sanity only.

---

## Context
Link every Source-1 (S1) business to all of its Source-2/3 records using only name, address and country.
Two outputs: `matching_results.tsv` (scored) and `candidate_pairs.tsv` (blocking audit), plus a reproducible
code package and a filled `Documentation_template.md`. Goal: a top-rank macro F0.5 with a pipeline that holds up
when reviewers audit it.

### Challenge card
- **Metric**: macro F_beta, beta = 0.5 (README; the video says 0.4, so I also report F0.4). It is computed per S1
  entity and then averaged. A true singleton predicted empty scores 1, and predicting any match for one scores 0.
  An entity that has matches but gets an empty prediction scores 0.
- **Scale**: the train set has 2.21M S1, 5.03M S2 and 5.29M S3 records. The test set has 1.73M S1, 4.89M S2 and
  5.08M S3. This is a big-data blocking problem.
- **Labels**: 5.6% of S1 entities are singletons, and the mode is 3 to 4 matches. There are 7.64M links, and **every
  S2/S3 id links to at most one S1 id**, a strict one-to-one constraint. About 2.7M S2/S3 train records are unlinked
  distractors.
- **Countries**: train has US 60% and India 40%. Test has India 47%, US 38% and **France 15%, which never appears in
  train**. Test also has more S2/S3 records per S1 (5.8 against 4.7), so it has more distractors and false merges
  cost more.
- **Noise** (from inspecting samples):
  - Hindi, Tamil and Kannada transliterations of English names ("होटल एंटरप्राइजेज लिमिटेड").
  - State names in native script or abbreviated (UP, TN, "தமிழ்நாடு", NY vs New York).
  - Domain-style names (`maurewilliamscolombier.com`) and unrelated DBA names ("Dréxkor").
  - Typos, accents, junk prefixes (`--`, `<<`), phone numbers, `null` tokens, reordered address components.
  - Leading zeros in house numbers (AF-0684), ordinals (45ND/45th), and about 3% empty addresses.
- **Rules**:
  - No external lookups, APIs or geocoding.
  - Models must be MIT or Apache-2.0 and at most 8B parameters.
  - 5 submissions per day and 3+ days left.
  - Use pip/HF libraries and weights only, never business data.
- **Compute**:
  - Work runs through SLURM partition `mtech`, never `test`. The QOS allows at most 2 GPUs and 2 jobs per user.
  - Nodes cn02–05 have 32 CPU cores, 240 GB RAM and 2 GPUs each. cn07 has 64 cores and 6 GPUs.
  - The login node is only for editing, submitting jobs and pip installs.
- **Storage**: `/csehome` is 100% full (29 GB free). All data, env and HF cache go in `/scratch/m25csa019`
  (200 GB quota). Code stays small, in `~/amazon_ml/er/`.
- The skill's tabular scripts do not fit pairwise ER. I reuse only its discipline: shared folds, OOF-only
  decisions, an experiment log, Optuna, and the write-up template in `references/playbook.md`.

## Step 0 (first action after approval)
Copy this plan to `~/amazon_ml/er/PLAN.md` and add a **Status** checklist at the top (one box per pipeline stage /
lever, with date + OOF when done). Update it after every stage so any session can see where we are.

## Layout
```
~/amazon_ml/er/                      # becomes code/business_entity_resolution/
  src/metric.py     macro F_beta exactly as README (+ unit test: README example = 0.714)
  src/norm.py       unicode/accents, junk strip, legal-suffix + street-abbrev canon (US/IN/FR),
                    script detection, transliteration, address parse (house no, unit, pin/zip, city, state)
  src/dicts.py      mine Indic->Latin token dict + state/city alias map from matched train pairs
  src/prep.py       6 TSVs -> normalized parquet (/scratch/m25csa019/er_work/)
  src/embed.py      GPU: multilingual-e5 embeddings (name, addr, full) -> float16 npy
  src/block.py      multi-pass candidate generation, union, light pre-ranker cap, recall report
  src/feats.py      pair features (rapidfuzz, multiprocessing 32 cores)
  src/train.py      LightGBM, 5-fold GroupKFold by S1 id, stage-1 + stage-2
  src/decide.py     one-to-one assignment, expected-F0.5 set selection, singleton gate
  src/ce.py         cross-encoder train/infer on uncertain band (lever)
  src/predict.py    test end-to-end -> output/matching_results.tsv, candidate_pairs.tsv
  jobs/*.sbatch     one per stage (partition=mtech, -c 32 --mem 200G, --gres=gpu:1 only when needed)
  run_all.sh, README.md, requirements.txt, EXPERIMENTS.md (log: exp | change | OOF F0.5 | per-country | LB | kept)
```
Env: `conda create -p /scratch/m25csa019/envs/er python=3.11`, then install polars, pyarrow, rapidfuzz,
lightgbm, faiss-gpu (or torch-based kNN), torch, sentence-transformers, unidecode, indic-transliteration, optuna,
jellyfish and scikit-learn. Set `HF_HOME=/scratch/m25csa019/hf`. The first GPU job runs `nvidia-smi` to record the
GPU type.

## Pipeline

### 1. Normalization (`norm.py`, `dicts.py`)
- **Base clean-up**: NFKC, lowercase, `unidecode` for Latin accents, strip junk punctuation, phone numbers and `null`.
- **Legal forms**: canonicalize them and keep them in a separate field from the core name:
  - pvt/private, ltd/limited, llp, llc, inc, corp, co/company
  - sarl, sas, sasu, sci, eurl, sa, cie
- **Street abbreviations**:
  - US: st, ave, rd, dr, ln, blvd, ct, trl, ter
  - France: r/rue, av/avenue, bd/boulevard, pl, all/allée, chem, imp
  - Ordinals and leading zeros in house numbers are normalized too.
- **Learned transliteration** (legal because it is mined only from provided data):
  - Align native-script tokens with S1 English tokens in matched pairs of equal token count to get a
    token→English dict ("प्राइवेट"→private, "राम"→ram).
  - Mine state/city aliases the same way (UP↔uttar pradesh, TN, native-script state names, NY↔new york).
  - Fallback for unseen tokens: indic-transliteration → Latin → consonant-skeleton phonetic key.
- **Address parse**: house/door number(s), unit, 6-digit PIN (India), street tokens, city and state. Every component
  is optional.
- **Domain names**: remove the TLD and compare against the S1 name concatenated without spaces.

### 2. Blocking (`block.py`): recall ceiling first
- All passes run within the same country (country is used as a filter only, never as a model feature). The passes:
  1. Dense kNN. e5 embeddings of `name | address` and of name only, exact top-k inner product on GPU, in both
     directions (S1→S2/S3 and S2/S3→S1, because each S2 has one true S1).
  2. Char-3gram TF-IDF top-k on the normalized, transliterated core name (chunked sparse matmul).
  3. Rare-token keys. The two highest-IDF name tokens and the two highest-IDF address tokens, each with city or state.
     Block size is capped at about 200.
  4. Address keys: (house no + first street token), (PIN + house no), (street + city). These catch DBA-name matches.
  5. Name-only keys for empty-address records, and concatenated keys for domain names.
- **Union and cap**: take the union, then apply a cheap LightGBM pre-ranker on about 10 fast features and cap at
  top-K per S1 (K tuned, about 30–60). The output of this cap is `candidate_pairs.tsv`, the exact set the main model
  scores.
- **Reports on train**:
  - pair recall
  - entity-level ceiling (macro F0.5 of a perfect classifier restricted to the candidates)
  - candidates per S1
  - all three broken down by country
- **Target**: pair recall ≥ 98% at ≤ 60 candidates per S1.
- **Dev subset for fast iteration**: all records of a set of states (about 15% of data). This keeps distractor
  density realistic.

### 3. Matching model (`feats.py`, `train.py`)
- **Pair features**, all country-agnostic so they transfer to France:
  - **Name**:
    - rapidfuzz ratio / partial / token_set / token_sort, Jaro-Winkler, token Jaccard
    - IDF-weighted soft token overlap, char-3gram cosine
    - core-name vs legal-form agreement or conflict, acronym match, domain-concat match
    - script flags, name-frequency (generic names like "Dental Clinic" are risky)
  - **Address**:
    - house-number match / conflict / missing (a key feature)
    - PIN/zip match or conflict, street sim, city sim, canonical-state match
    - unit match, token_set, IDF overlap, missing flags
  - **Embeddings**: cosine for name, address and full text.
  - **Meta**: source (S2/S3), candidate block origin flags, block sizes.
- **Stage 1**: LightGBM binary. OOF comes from 5-fold GroupKFold by S1 id (valid because every S2/S3 has at most one
  S1). Training uses a large S1 sample, about 600k S1 or ~30M pairs, and all train S1 are scored OOF.
- **Stage 2**: the stage-1 features plus competition and context features, all computed from stage-1 OOF scores:
  - rank and gap to the best score within the S1's candidates
  - the S2/S3's reverse rank among all S1s, and the margin to its second-best S1
  - "support", the max similarity to that S1's other high-confidence matches. This helps DBA-name records that share
    an address with an accepted sibling.
- **Robustness check for France**:
  - Leave-one-country-out: train on US and score India, then the reverse.
  - Keep only features and parameters that hold up under LOCO.
  - Watch the France test score distributions for drift.

### 4. Decision (`decide.py`): metric-aware post-processing
1. **One-to-one**: each S2/S3 is kept only for its argmax S1.
2. **Per-S1 set selection**:
   - Sort the calibrated probabilities and pick the top-k set, or the empty set, that maximizes expected F0.5.
   - The empty set's value is P(singleton), which comes from a small entity-level model on max score, count above
     cutoffs and candidate count.
   - Compare against a global threshold on OOF and keep the winner.
3. Tune on OOF. Report macro F0.5 overall and by country, and F0.4 as a side check.

### 5. Levers in order of expected payoff (one at a time, logged in EXPERIMENTS.md)
1. Blocking recall passes. The ceiling limits everything else.
2. Learned transliteration and alias dicts.
3. Stage-2 competition features and one-to-one assignment.
4. Expected-F decision and singleton gate.
5. Cross-encoder (multilingual-e5-base, MIT, or xlm-roberta-base, MIT):
   - Fine-tune on hard pairs.
   - Score only the uncertain band (p in 0.05–0.95) with 2 GPUs.
   - Feed its OOF score in as a GBDT feature.
6. Self-training on test. Add high-confidence one-to-one test pairs, including France, as pseudo-labels and refit.
   This uses provided test data only, never test labels.
7. Optuna tuning of LightGBM, then a seed average.

A row-order or ID leak would not be used, even if found, because the top packages are audited.

## Timeline (5 subs/day, submit only meaningful jumps)
- **Day 1**:
  - env, prep, dicts, blocking v1 with recall report on dev
  - stage-1 features, LightGBM and threshold
  - full test run → **first submission** (safe baseline kept)
- **Day 2**: full blocking passes, stage 2, expected-F decision, singleton gate, LOCO checks → 1–2 submissions.
- **Day 3**: cross-encoder band, pseudo-labels, Optuna, seeds → submissions. **Freeze new ideas about 18h before the
  deadline.**
- **Final**:
  - clean end-to-end rerun from raw TSVs via `run_all.sh`
  - validator PASS
  - zip package (`output/`, `code/business_entity_resolution/`, filled `Documentation_template.md` with ablation
    table, model licences and parameter counts)

## Verification
- `python src/metric.py` has a unit test: the README example must give 0.714. It also checks the singleton rules.
- Blocking report on the train dev subset and on full train: pair recall, entity-ceiling F0.5, candidates per S1.
- The OOF macro F0.5 printed after every stage (overall, US, India, LOCO) is recorded in EXPERIMENTS.md, and every
  recommendation cites it.
- The test output is checked with the official validator from `student_resource/`:
  `python3 utils/validate_submission.py --matching <out>/matching_results.tsv --candidate <out>/candidate_pairs.tsv --test-dir dataset/test`
  It must print PASS, with no warnings about matches that are not candidates. `--check-ids` is run once in an sbatch
  job because it needs a lot of memory.
- Sanity checks: 1,732,544 rows, a predicted-singleton rate per country close to the OOF rate, no S2/S3 id
  assigned to two S1s.


---
# Phase 2 plan (approved 2026-09-25)
# Improvement plan after w2: Business Entity Resolution (Amazon ML Challenge 2026)

## Context
- **Best version:** w2, OOF macro F0.5 **0.98634** on the test-like `tsim` split (US 0.98599, India 0.98686). Submission at `er/submissions/w2/`.
- **The user asked for detailed research into further gains and a plan**, without overfitting.
- **The previous plan and its status** live in `er/PLAN.md`. Versions are tracked in `er/VERSIONS.md` and the ablation log in `er/EXPERIMENTS.md`.
- **w2a (just finished):** w2 trained on 50% of S1 entities instead of 15%, lr 0.05.
  - **OOF 0.98670** (US 0.98643, India 0.98711), **+0.00036 over w2**.
  - Evidence that more training data helps, which motivates a cascade with 100% of the data.
- **Running:** w3a = w2a plus the cross-encoder trained on the kx band (cew, AUC 0.974).
- **No leaderboard feedback.** The upload is stuck on an organiser-side error. OOF on `tsim` (plus the audits) is the only yardstick, so the noise-floor rule below is essential.
- **Timeline:** about 2.5 days remain, so the full plan applies.

### Where the remaining loss is (evidence)
- **Total loss** is about 0.0137 × 1.79M S1 ≈ 24k.
- **Candidate-set ceiling** (kx = K100 + sibling expansion) is 0.9936.
  - About **47% of the remaining loss is blocking**: 118k true pairs never become candidates.
  - About **53% is the model and decision**.
- **The uncapped `tsim` blocking ceiling is 0.9940** at 160 candidates per S1, against 0.9928 at K100.
- **u3 error split:**
  - missed links: 65% of loss, of which 63% blocked
  - matched entities predicted empty: 20%
  - wrong merges: 11%
  - singleton false merges: 4%
  - precision 0.9977, recall 0.9595
- **Decision rule has plateaued:** 48 settings give 0.98519 vs 0.98517. It's not a lever.
- **w2's stage-2 gain:** c_p 0.78, c_marg2 0.15, then the cross-encoder features.
  - Stage 2 is essentially "stage-1 probability plus competition".
  - Collective (sibling) evidence is not used anywhere yet.
- **Cascade sizing** (w1 OOF on 172.4M kx pairs):

  | cutoff p1 | pairs kept | true pairs lost (of 6.07M) |
  |---|---|---|
  | ≥ 1e-3 | 8.0M | 778 |
  | ≥ 1e-4 | 10.2M | 76 |

  The pairs lost are never selected anyway, since the decision floor is 0.02.
- **Constraints:**
  - w2 peaked at 221 GB RAM (limit 235) and stage 1 on kx at 213 GB. Nothing bigger fits without streaming.
  - QOS allows 2 GPUs and 3 submitted jobs. `/scratch` has 110 of 200 GB used.
  - About 2.5 days remain; 5 leaderboard uploads per day.
- **Audit (already done):**
  - Train→test drift is mild (adversarial AUC 0.64 US, 0.59 India).
  - France `st_eq` gap costs about 0.0001 on the leaderboard.
  - Transliteration dictionary coverage is 97.8% on train vs 96.4% on test.
  - Calibration is good.
  - None of these justify a fix now.

## Guard rails (apply to every step)
- **Folds:** always the same 5 folds (hash seed 7) and the same `tsim` split. Every stacking input is OOF:
  - ctx and support features come from OOF stage-1 scores
  - the cross-encoder is 2-fold by S1
- **Noise floor first:** train w2's config with 2 seeds.
  - Accept a lever only if its OOF gain is > max(0.0001, 2 × |seed difference|).
  - One lever per experiment. Log every run in `EXPERIMENTS.md`.
- **Leaderboard is a sanity check only.** Upload when OOF improves by ≥ 0.0003. Keep w2 as the safe backup.
- **Release path for any winner:**
  - `jobs/release.sh` `release_cmds` (apply → validator → `snap.py`) → `submissions/<tag>/` and `VERSIONS.md`.
  - The w2 family must pass `cand=cand_test_kx`, so `candidate_pairs.tsv` = the scored set.

## Steps (ordered by gain per hour)

### 0. Close running work and measure (≈1 h, cheap)
- **Release w2a now** (`release_cmds w2a …` with `cand=cand_test_kx`) as the new best submission: 0.98670 vs 0.98634.
- **Collect w3a's OOF.** Adopt cew only if it beats w2a by more than the noise floor.
- Run `ana/err.py tag=w2 split=tsim` for the current loss split.
- Run a second seed of the best config to set the noise floor.

### 1. Cascade stage 2 (enabler, ≈2 h) — `src/train.py`
- **New option `sub=<oof_tag>:<pte_tag>:<thr>`:**
  - While reading each feature part, semi-join to pairs with stage-1 p ≥ thr (1e-3), then join extras.
  - Peak memory drops from 221 GB to about 20 GB.
  - OOF and pte cover only the subset. `decide.py` works unchanged, because the one-to-one max and the ef floor 0.02 never select sub-cutoff pairs. Check this once.
- **Reuse the existing pieces:** `load()`, `DROP`, `fold_of()`, the seed/lr options, and `blend.py` for seeds.
- **Run on the cascade:** frac 1.0, 3 seeds, blended with `src/blend.py` (logit mean, weights on OOF).
- **Expected:** at least w2a's level. The main gain is that everything after this becomes cheap.

### 2. Collective "sibling support" features (≈2 h) — new `src/sup.py`
- **Siblings of an S1:** its confident matches from w1 OOF (tsim) or w1 pte (test): p ≥ 0.9, one-to-one via `decide.one2one`, excluding the pair itself.
- **For each cascade pair (s1, x):**
  - number of siblings
  - max and mean normalized-name similarity (token_set, JW) between x and the siblings
  - max address similarity and a same-first-house-number flag
  - **raw-string** exact / ratio matches for name and address. Siblings often share identical native-script text or source formatting, which our normalization can lose.
  - max(min(name, addr)) over siblings
- **Why it should help:**
  - Partial-recall entities are 65% of the loss, and they have confident siblings by definition.
  - It also catches distractors that differ from the siblings.
- **Implementation:** copy the rapidfuzz and Pool pattern from `src/feats.py` (`_chunk` / `pair_row`). The input is about 8M pairs × ~3.4 siblings, a few minutes on 32 cores.
- **Wiring:** add as `sup_{split}` to the cascade stage-2 extras.

### 3. Second, multilingual cross-encoder on raw text (GPU, ≈2.5 h, can overlap steps 1–2) — `src/ce.py`
- **Model:** `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (Apache-2.0, 118M parameters; licence checked via the HF API).
- **Input:** raw names and addresses (native scripts, case, punctuation), formatted as "name: … | address: …".
- **Training:** 2 epochs, the w1 band, 2-fold OOF.
- **Code changes:**
  - `texts()` gets a `txt=raw` option. The `n_*` parquets already hold the raw `name` / `addr` columns.
  - The output column name comes from `name=`.
- **Why:** a diverse error source that sees what normalization discards. Add as feature `cem`.
- **Fallback if throughput is too low:** `BAAI/bge-reranker-base` (MIT, 278M parameters), which is about 3× slower.

### 4. Candidate recall (≈4–5 h) — needs streaming
- **Streaming stage 1 in `train.py`:**
  - Fit only on sampled-S1 rows (read per part).
  - Predict OOF and test per part.
  - A one-time merge step writes extras into partition-aligned parts, using the same `hash(id1, seed=3) % n` layout as `feats.py`.
  - This removes the 213 GB peak.
- **Levers, in order:**
  - **(a) Expansion round 2:**
    - `expand.py` gets an `out=` option. Run with pred = w2 (or the step-1 cascade) and base = `cand_{split}_kx`.
    - Measure the recovered fraction of the remaining 118k missed pairs first. That is cheap and the script already prints it.
  - **(b) K200 caps** via `ext.py make k1=200 k2=15` (uncapped ceiling 0.9940). Features only for new pairs (`feats.py`), then `ext.py link`.
  - **(c) Optional probe:** a char n-gram / e5-small ANN blocking pass. Go only if it recovers ≥ 15% of remaining misses at ≤ 10 new pairs per S1.
- **Rule:** adopt only if the ceiling rises by ≥ 0.0005. Rebuild xf / f2 / sib on the new set, then run stage-1 → CE band → ctx / sup → cascade stage 2.

### 5. Model diversity and tuning on the cascade (≈2 h, cheap after step 1)
- **Optuna:** 30 trials of LightGBM stage-2 parameters (num_leaves, min_data_in_leaf, feature_fraction, lambda_l2) on 2 folds with logloss, then refit on 5 folds.
- **XGBoost:** `hist`, GPU, same features, as a second family. Blend both with `src/blend.py` (grid of weights on OOF).
- **Optional entity gate:** an S1-level LightGBM for P(no match), built from aggregated pair scores plus name commonness, empty address and candidate count (OOF). Use it in place of Π(1−p) in `decide.pick_ef`. It targets the 24% of loss from "predicted empty / singleton false merge".

### 6. Consolidate, freeze and package (last ≈18 h)
- **Freeze ideas about 18 h before the deadline.**
- **Final reproducible chain in `er/run_all.sh`:**
  1. prep / dicts / norm and mksim
  2. blocking → K100 (→ K200 if adopted)
  3. feats / xf / f2
  4. stage-1 A → expansion → kx
  5. stage-1 B (streaming)
  6. cross-encoder(s) on B's band
  7. ctx + sup
  8. cascade stage 2 (seeds + XGBoost) → blend → decide
- **This drops the legacy t1 / u4 / v1 dependencies.** Verify its OOF ≥ best − noise; otherwise script the legacy chain exactly.
- **Deliverables:**
  - Run end to end from the raw TSVs; validator PASS (plus `--check-ids` once).
  - Final `snap.py` release.
  - Zip: `output/` (matching + candidate_pairs from the final scored set), `code/business_entity_resolution/`, `Documentation.md` with the final ablation table and model licences.

## Expected gains (honest ranges, OOF on tsim)
| step | lever | expected Δ F0.5 |
|---|---|---|
| 1 | cascade + full data + 3 seeds | +0.0002 to +0.0006 |
| 2 | sibling support features | +0.0003 to +0.0010 |
| 3 | multilingual raw-text cross-encoder | +0.0002 to +0.0005 |
| 4 | expansion round 2 + K200 | +0.0003 to +0.0008 |
| 5 | Optuna + XGBoost blend + gate | +0.0002 to +0.0005 |

Reaching 0.988–0.989 OOF is plausible. None of these gains are guaranteed; each is kept only if it clears the noise floor.

## Critical files
- **Modify:**
  - `er/src/train.py` (sub / stream / seed)
  - `er/src/ce.py` (raw text, output name)
  - `er/src/expand.py` (out=)
  - `er/src/decide.py` (optional gate)
  - `er/run_all.sh`
  - `er/Documentation.md`
  - `er/PLAN.md`, `er/EXPERIMENTS.md`
- **New:**
  - `er/src/sup.py`
  - `er/src/merge_parts.py` (extras → aligned parts, for streaming)
  - `er/src/xgb.py` (or an `algo=xgb` switch in `train.py`)
- **Reuse:**
  - `block.enc / cap / gen_pairs`
  - `ext.py`, `kx.py`
  - `ctx.py`
  - `decide.one2one / select / score`
  - `blend.py`
  - `snap.py`, `jobs/release.sh`
  - `jobs/multi.sbatch`, `jobs/gpu.sbatch`
  - `ana/err.py`, `ana/calib.py`, `ana/audit.py`

## Verification
- **Every step:**
  - `decide.py tune tag=<t> split=tsim` prints OOF (ALL / US / India / F0.4 / pred_empty).
  - Compare against w2 and the noise floor, and log the result in `EXPERIMENTS.md`.
- **Cascade parity:** decide on the subset vs the full pairs must give the same F0.5 for w1/w2 predictions, within 1e-5.
- **Recall steps:** `expand.py` / `block.py report` print the recovered fraction and ceiling before any retraining.
- **Winners:** `release_cmds` → validator PASS → `submissions/<tag>/matching_results.tsv` plus a `VERSIONS.md` row. Upload for a leaderboard sanity check.
- **Final:** clean `run_all.sh` rerun from the raw TSVs reproduces the chosen version's OOF and test output (validator PASS). Then build the zip.
