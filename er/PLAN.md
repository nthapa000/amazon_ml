# PLAN — Business Entity Resolution (Amazon ML Challenge 2026)

## Status (update after every stage: date | result | OOF F0.5)
Best so far: c5 OOF 0.99077 (tsim) -> submissions/c5. LB: c2 0.982, c6 0.9835 (France ~0.955, US+India ~0.9885). Versions + submissions: VERSIONS.md, submissions/<tag>/matching_results.tsv, /scratch/m25csa019/versions/<tag>/
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
- [x] 12. France robustness: superseded by adversarial audit + France EDA fixes (P11)
- [x] 12b. tsim split (19% S1 dropped) to mimic test orphan rate | 2026-09-25
- [x] 13. Cross-encoder on uncertain band (MiniLM-L6 -> u2 OOF 0.98422, sub3 submissions/u2); L12 wide band (ce12) -> u3 running
- [-] 14. Pseudo-label self-training: skipped (France fixed via normalization instead; unmeasurable, risky)
- [-] 15. Optuna + seeds: seeds useless (noise 0.000004); Optuna skipped (low expected gain)
- [~] 16. Final end-to-end rerun running (P12); zip + Documentation pending

## Phase 2 status: improvement plan after w2 (full plan: section "Phase 2 plan" below)
- [x] P0. w2a released (OOF 0.98670, sub7); w3a 0.98676 (+0.00006, not adopted); err analysis w2 done; cascade parity exact; noise floor -> from c1a/c1b
- [x] P1. Cascade stage 2: c1a 0.98685 (+0.00015 vs w2a), released; seed noise 0.000004
- [x] P2. Sibling support: c2s 0.98714 (+0.00029)
- [x] P3. Multilingual CE: c2m 0.98757 (+0.00072); combined c2 0.98776 (+0.00091) released
- [x] P4. Recall: NO-GO (expansion round 2 recovers 0.6% of misses; K200 +0.0004 < bar)
- [x] P5. XGBoost c2x 0.98778 / blend c2+c2x 0.98779 / gate +0.00001 -> none adopted; Optuna skipped
- [~] P6. Consolidated run = P12 (running); package zip after it
- [x] P7. bge-reranker-base raw-text CE (band AUC 0.9804) -> c6 = c2 + ceb: 0.98790 (+0.00014), released
- [x] P8. Error review of c2 -> c4 (prefix-ambiguity + source-aware siblings): 0.98783 (+0.00007, below bar, not adopted)
- [x] P9. Gate +0.00001, XGBoost 0.98778, blend 0.98779 -> none adopted
- [x] P10. ANN: c5 OOF 0.99077 (+0.00287 vs c6) released. Probe GO: top-3 recovers 43.5% of blocked misses (51.4k links) -> integrating (chain: ann test -> ka features -> w1ann -> ctx/sup -> cemka -> c5)
- [~] P11. FRANCE (scope chosen: fixes + France-only LB threshold probes; no pseudo-labels; geography rule-based table, exactly as used by fin): LB 0.982 vs OOF 0.9878 -> France normalization fixes (dept->region, N°, hyphens, ordinals, legal) in consolidated run 'fin'; probe c6_noFR prepared. (was: Optional, guarded: France adaptation (pseudo-labels from confident test pairs incl. France) and test-side transliteration-dict extension; accept only if tsim OOF does not drop
- [x] P12. Consolidated end-to-end run from raw TSVs started 20:55 (jobs/final.dag -> er_final), ETA ~09:00-10:00 Sep 26; then final zip
  - Sep 26 02:33: base, a1, cev, kx, ann, annte, b1 DONE; ka FAILED in `annka.py union split=test` (/scratch quota: 201G of 200G soft / 205G hard) -> cem, ceb, fin skipped.
  - Sep 26 02:48 (user chose option B): deleted 24G of consumed er_final intermediates (cand0_*, cand_{tsim,test}, *_k100new, xf/f2_*_k100, oof/pte_a1) -> 177G; union + ctx now write only rows with stage-1 p >= 0.0005 (stage 2 reads p >= 0.001; synthetic equivalence test passed). Resumed: `jobs/resume.dag` (ka2 41217 -> cem, ceb -> fin), log logs/dag_resume.log, release watcher `jobs/after_fin.sh logs/dag_resume.log`. fin ETA ~8 h after 02:48.
  - 02:50 and 02:58: ka2 failed again with 'Disk quota exceeded' at 177G, always on xf_test_ka.parquet (the file being written at the real 02:33 quota hit): the NFS client on cn07 kept re-reporting the old error when that file was rewritten in place (new files wrote fine; cn07 /tmp fine). Fix: annka.py union writes to <file>.tmp then renames. 02:59 relaunched (ka2 41221, logs/dag_resume3.log, watcher after_fin.sh logs/dag_resume3.log); union test OK 03:0x.
  - 03:24 ka2 DONE; 04:49 cem DONE (band AUC ce 0.978, RTX A6000 on cn07). ceb FAILED 03:26: landed on cn07's RTX PRO 6000 Blackwell (sm_120), unsupported by our torch -> fin skipped (not noticed until 10:59). 11:00 re-run ceb pinned to cn05 A30 (41235, jobs/resume2.dag, logs/dag_resume4.log); ~5 h -> fin ETA ~17:30-18:00.
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
- [x] Package docs (2026-09-26 11:5x): README.md rewritten for the final pipeline (11 steps, models + licences, SLURM/one-machine run); run_all.sh generated from jobs/final.dag (52 commands, no personal paths); jobs/env.sh paths overridable (same defaults); requirements.txt pinned from the env.
- 2026-09-26 15:06: fin DONE + released (submissions/fin). OOF 0.99089 (US 0.99048, India 0.99152, F0.4 0.99257) vs c5 0.99077; validator PASS; vs c5: 25,772 French rows changed (9.9%), 5,054 India, 4,242 US. Awaiting LB upload by user.
- [~] Improvement round (user choice 2026-09-26 ~15:30, 12-24 h left): Optuna on stage 2 + 4th cross-encoder BAAI/bge-reranker-v2-m3 (568M, Apache-2.0, 'cer'). jobs/improve.dag, log logs/dag_improve2.log. ce.py save=0 option (no weights; disk). 15:46 opt (cn02) + smoke (A30 bs64 fits, 126 train pairs/s); 15:53 cer on cn05 A30, ETA ~21:30. Variants: fin2b (tuned params), fin2a (+cer), fin2 (both); keep a lever only if OOF gain > 0.0001.
- [x] Candidate-size rule (organizers' update 2026-09-26): final candidate set = stage-1 learned filter p1 >= 0.005 (user choice), src/cands.py; finc released (4.47 cand/S1, OOF 0.990853, PASS); final.dag + run_all.sh (jobs/gen_run_all.py) + README + Documentation updated. Apply the same step to whichever final model wins (fin2a/fin2 after cer).
- 17:45 fin2b adopted after unbiased folds-1-4 check (+0.000020); fin2bc released (OOF 0.990870, 4.47 cand/S1, PASS) = current best. Next: fin2 (cer + tuned) vs fin2bc tonight; final.dag gets opt + pfile (+ cer if adopted) once the winner is known.
- 19:50 fr=-2 probe LB 0.986946 (+0.000018 vs fr=-1; marginal links ~23% wrong = break-even) -> fr=-2 adopted. LLM judge smoke test: AUC 0.531 vs 0.772, no stacking gain -> dropped (with the France rule). Remaining: cer final model (upload B) + fr=-2, then freeze + package.
- 20:45 fin2 (Optuna + cer) OOF 0.990957 (+0.000049 vs fin2b; 4/5 folds +); upload B = probe_fin2_frm2 ready (fin2 + fr=-2, PASS). 20:58 cer2 launched (user): bge-reranker-v2-m3 on ~1.6M band pairs per half x 2 epochs, halves as parallel GPU jobs (jobs/cer2.dag, logs/dag_cer2.log) -> merge -> fin3 (fin2 with cer2) -> cmp vs fin2. cer outputs kept as reference; adopt cer2 only if fin3 beats fin2 on folds 1-4.
- 22:00 cer2 moved to cn07 RTX PRO 6000 Blackwell GPUs: cn03's A30 was hardware-throttled (HW slowdown + power brake, 232 MHz), all other A30s busy. Side env /scratch/m25csa019/envs/erbw = venv over envs/er + torch 2.7.1+cu128 (sm_120); original env untouched. Both halves 8.8 steps/s (A30 3.15); ETA cer2 ~00:05, fin3 + fold check ~00:45 (jobs/cer2bw.dag, logs/dag_cer2bw.log).
- 23:20 Final-day plan (user approved; this session executes, the other session only analyses): L4 unseen-country shift done; L1 French noisy student running (jobs/cef.dag: smoke -> 2-GPU cef, ceb warm start + fra_b1ka + US/India replay) -> stage 2 with cef replacing ceb -> fold check -> 'c' release + fr=-2 -> upload. Uploads: 6 left; upload 1 = package-consistent winner of fin3/fin2 ('c' + fr=-2); B (fin2_frm2, full set) NOT to be uploaded. Freeze Sun ~16:00, zip by ~20:00.
- 00:55 fin3 adopted (+0.000159 OOF, all folds). Upload 1 = fin3c_frm2 (release job 41396). ceq 4B run launched (1.2M x 2 epochs, 2 Blackwells in one job). fin4 = fin3 + cef (replaces ceb) after cef merge. Final deadline 20:00.
- 27 Sep 02:15: /csehome full (shared FS, 30 MB free). Moved old submission folders (c1a..w2a, probes except probe_fin_frm2) to /scratch/m25csa019/sub_archive (verified copies).
- 05:20 `ce.py` fix (before `ceq` finishes, so it's ready the moment `ceq_b1ka_h{0,1}` save): `init=` reload of a saved
  LoRA adapter was silently frozen (transformers 5.17.0 auto-loads it via `AutoModelForSequenceClassification.from_pretrained`
  with `is_trainable=False` hard-defaulted, then the old `if lora:` line stacked a second, blank adapter on top instead of
  resuming it -- no crash, no error, just wrong). Fixed: detect a saved adapter dir, reload via
  `PeftModel.from_pretrained(base, dir, is_trainable=True)` instead, skip `get_peft_model` for that case. CPU-only mechanism
  test (scratchpad, tiny cached model, no GPU -- both held by `ceq`) confirms: old path reloads `requires_grad=False`; new
  path reloads `requires_grad=True` with weight values identical to the trained adapter (not reinitialized). Ready for the
  ~11am `ceq_b1ka_h{0,1}` -> Kushal handoff (gated on fin4's LB result per the plan).
- 27 Sep 11:05 (user: work until 22:00, no 16:00 freeze; run the French 4B `ceqf` regardless, side by side with fin5; 4 uploads left;
  team best public = Kushal 0.987862). Pipeline: `ceq` (job 41397) -> `jobs/fin56.dag` (runner started 11:02, logs/dag_fin56.log):
  w5 waits for 41397 -> f5 (`jobs/par.sbatch`: fin5a = fin3 + ceq and fin5b = fin4 + ceq in parallel, blend -> fin5, fold check vs
  b34, release fin5c / fin5c_frm2) -> w6 waits for `ceqf_b1ka_test` -> f6 (fin6 = fin5's saved fold models re-scoring test with
  ceqf patched into unseen-country pairs: `train.py models=<tag> alias=ceqf:ceq`, no retraining; release fin6c / fin6c_frm2;
  `ana/diff_country.py` must show only French rows changed). ceqf = GPU job via jobs/bw_retry.sh (gpu2bw.sbatch):
  init=ceq_b1ka, 200k fra_b1ka + 150k US/India replay, lr 2.5e-5, 1 epoch, unseen=1 (908,542 French test band pairs),
  merge unseen=1 base=ceq. Launch after the real-adapter load test (~11:30).
- 27 Sep 11:17 prep done. (1) `train.py models=<tag> alias=new:old` predict-only re-scoring: reproduces fin3's test predictions exactly
  (max |dp| 0.0, 11 min; job 41505) -> fin6 ~40 min after ceqf. (2) ceq real-adapter load test (CPU): 33M trainable (LoRA + score
  head), saved tensors reload unchanged, trained head active. (3) ceqf queued (41515, pending on GPU quota behind 41397), reads
  `fra_ceqf.parquet` -> symlink to `fra_b1kb` (user: use b1kb if ready; other session's set: same neg/s1neg rows, pos = independent
  resample of the same pool, syn with generic last-word changes). `fradapt.py lw=0` = original noise (cef's fra_b1ka).
  (4) final.dag: 22 steps (opt, cer, fin2, fra, fra2, cef, cer2, ceq, fin5, ceqf, fin6, out), run_all.sh regenerated (74 commands);
  README (models, pinned HF revisions, torch 2.7.1+cu128 + peft, steps 11-22), requirements.txt, Documentation.md sections 4/5
  updated; final numbers marked ⟨TBD⟩ (fill after fin5/fin6 + LB). If the final is fin5 (not fin6): final.dag `out` -> fin5.
- 27 Sep 18:58 ceq DONE (band AUC 0.98563, best CE). f5 (fin5) + ceqf RUNNING on cn07 since 18:55 (ceqf restarted as 41942 with
  4 CPU / 140 GB so f5 could fit; f5 mem 60G). Quick fixes from `plan (1).md` via new ana/qfix.py (no src/ edits): Q2 count
  prior / caps no gain; Q4 per-source France shift dropped; Q1 France shift re-fit -> probe later; Q5 sibling support from
  fin3 scores (sup3) + Q3 fin5+b34 blend queued as q_f5x (41943, afterok f5). Friend merge = last priority.
  Uploads plan: fin5c_frm2 (~20:10) -> Q5/Q3 winner (~21:20) -> fin6c_frm2 (~22:40) -> France probe; freeze 22:30-23:00.
- 20:30 Took over the night plan from session amazon-ml-a3 (user-agreed): deadline 23:59, only matching_results.tsv needed
  tonight (both TSVs still written), 5 uploads, upload only likely LB gains, new code in er/ana/ only.
  Step 1 (w6) moot: w6 failed to submit (test partition 1 h cap), f6 resubmitted manually as 41947 afterok:41942 (no slot used).
  Step 2 A1: new ana/pcdec.py (per-country decision rule grid on tsim, CV-honest estimate, A5 runner-up option; apply writes
  both TSVs + dec json). Synthetic test: base reproduces the global rule, apply reproduces tune exactly. Queued via a
  login-node watcher (logs/q_pc_watch.log) the moment q_f5x (41943) exits (mtech submit limit 3 reached).
  Released so far tonight: fin5c, fin5c_frm2 (pruned, 0.991231), fin5_frm2 (unpruned, 0.991276), all validator PASS.
- 20:55 LB fin5_frm2 (fin5 unpruned + fr=-2) = 0.987997 = NEW TEAM BEST (prev 0.987862, +0.000135). Implied France ~0.9663 (fin_frm2 0.9615, cx1 0.9639): ceq moved France ~+0.0024 over cx1 and ~+0.005 over fin.
- 21:05 A1/A5 not adopted (noise). Upload #2 = fin5s_frm2 (tsim +0.000022 over LB-best fin5, 5/5 folds). Next: f6 + q_f6s after ceqf (~21:50) -> fin6s_frm2 (#3), fin6s at matched fr (#4).
- 21:07 step 4 re-plumbed for speed: f6 41947 (fin6a+fin6b + fin6 chain, no longer needed since the upload line is fin5s) cancelled; q_p6b 41953 (fin5b models re-score test with ceqf -> fin6b) and q_p6sa 41954 (fin5sa -> fin6sa) run in parallel afterok ceqf 41942; watcher3 submits q_f6s (blend fin6s -> fin6s_frm2 unpruned + fin6sc_frm2 pruned, diff_country vs fin5s(c)_frm2, France diag on fin6s) afterok both. ETA fin6s_frm2 ~22:25.
- 21:15 L3 (plan (1).md IV.2, per-country stage 2) using the idle CPU slot while ceqf scores: new ana/pcsub.py writes country keep-lists (tsim US 4,928,843 / India 3,265,904 pairs; test US 3,205,383 / India 3,874,556) so src/train.py sub=oof_b1ka_<c>:pte_b1ka_<c>:0.001 trains fin5sa's config per country (same folds). q_l3 41959: pcus || pcin -> blend fin5s+pcus+pcin (fin5sp) -> cmp_folds vs fin5s. France keeps the pooled model. To free a submit slot the pending q_p6b/q_p6sa were cancelled; watcher4 submits one par job q_f6 when ceqf COMPLETES: fin6b || fin6sa re-score, then fin6s_frm2 / fin6sc_frm2 releases, diff_country, France diag, and fin6sp_frm2 (= fin6s + L3) for use only if L3 passes its gate.
- 21:30 L1 gate (ana/xlgate.py on fin5s, tsim): India true pairs latin 2,032,361 recall 0.9720 (blocked 1.53%, missed in cands 1.28%, false-link rate 0.14%) vs native-script 446,706 recall 0.9969 (blocked 0.21%, missed 0.10%, FP 0.03%) -> native-script India is the BEST-served slice; no matched tsim record has unknown native tokens (dictionary learned in-sample). Records with unknown native tokens: tsim 1.55% vs test 2.57% of India S2/S3 (+1.0 pt, all still read raw by cer2/ceq/ceqf) -> L1 FAILS the gate, not built.
- 21:55 N4 text-view TTA on ceqf (ana/ce_tta.py; GPU q_tta 41969): French band test pairs whose S2/S3 raw text has spurious accents / non-ASCII or an all-caps name (432,279 of 908,542 = 47.6%) re-scored by both ceqf halves on a folded view (unidecode + lowercase, both records); score = mean(raw, folded). Merge -> ceqft_b1ka_test; then patch alias ceqft:ceq into fin5sa/fin5b (predict only) -> fin7s_frm2 (France rows only change vs fin6s). cer2f (plan B1) impossible: cer2 was trained with save=0 (no weights).
- 21:58 L3 per-country stage 2: fin5sp (0.4 pooled + 0.3 US model + 0.3 India model) 0.991304 vs fin5s 0.991298 = +0.000006, folds +0.000015 +0.000011 +0.000019 -0.000018 +0.000004, US +0.000013, India -0.000004 -> NOT adopted (country worse, below bar). q_f7 41970 (afterok q_tta): fin7sa/fin7b predict with ceqft -> fin7s_frm2.
- 22:00 team merge running (q_mx 41971: friend cxQ + our fin6s). Team has 3 uploads left (friend: keep 1 for a final blend ~23:00). Candidates: mx6_frm2 (~22:25), fin7s TTA (~23:00), France shift probe (fr ~ -1.4).
- 22:13 mx6_frm2 released (friend cxQ + our fin6s, tsim 0.991528). 3-person blend q_tb3 running (cxQ + fin6s + fx).
- 22:20 q_tb3 (41973): kfx gate PASS (rows exact, 100% keys, labels identical; corr 0.9991 fin6s / 0.9982 kcxQ). wblend best seen weights cxQ 0.7 / fin6s 0.3 / fx 0.0 = 0.991528 (= mx6 exactly on US/India; fx adds nothing on tsim); France = equal thirds. tb3_frm2 validator PASS, 5.8M links; job then failed at snap (';' inside the desc split CMDS) -> redone in 41976. /csehome down to 2.5 GB: 10 superseded submission folders (b34c*, fin2*c*, fin3c*, fin4c*, finc) moved to /scratch/m25csa019/sub_archive with md5 check -> 4.6 GB free.
- 22:30 (user) Uploads left = 5 (not 3). q_f7 41970 running on cn07 since 22:23 (fin7b || fin7sa predict with ceqft -> fin7s_frm2). Ready: tb3_frm2, mx6_frm2, fin6s_frm2, fin6sc_frm2.
- 22:45 q_f7 41970 DONE: fin7s_frm2 released (validator PASS, submissions/fin7s_frm2). vs fin6s_frm2 only 969 France rows changed (0.37% of France; US/India 0). France diag at fr=-2: links/S1 vs seen -0.1199 (fin6s -0.1187) -> TTA effect ~nil; not worth its own upload.
- 22:50 (user) fin7s_frm1.5 released (q_f7r15 41979; 41978 failed: ER_WORK not set to er_final): PASS, 5,832,341 links; vs fin7s_frm2 France 6,253 rows changed, US/India 0. Test candidate set (pte_fin7s, p1>=0.001): 5.30/S1 (France 8.12, India 4.78, US 4.83); 3.36 matches/S1.
- 22:52 mx7_frm2 = friend cxQ 0.7 + our fin7s 0.3 (fin7s = fin6s + ceqf text-view TTA; weights re-tuned: same 0.7/0.3): tsim 0.991528 (= mx6), PASS, 5,830,392 links; vs mx6_frm2 only 124 French rows differ (US/India 0) -> practically identical to mx6; TTA's effect is diluted to 0.3 and touched only 969 French rows in fin7s. fin7s_frm2 vs fin6s_frm2: 969 French rows. A separate session/user also released fin7s_frm1.5 (fin7s, fr=-1.5) at 22:50 (jobs q_f7r15 41978 failed on ER_WORK, 41979 OK).
- 23:10 (session a3) Q&A-driven: candidate size is judged as TOTAL PAIRS, separate from LB; pruned later stage allowed; private = best public. Built mx7c_frm2 (mx7 on p1>=0.005 cands: 7,739,003 pairs vs 9,187,520, -15.8%): tsim 0.991481 vs mx7 0.991528 (-0.000047), 664 rows differ, validator PASS. Queued tb3c_frm2 (41981) and mx7c_frm1.5 + France diag (41982). Recommend uploading only pruned versions so the best public carries the smaller candidate set. FR_DEP = hand-written department->region list (no postal/city) -> compliant.
- 23:26 Friend's blQF (cxQF+cxQ2+cxQ3 seeds, France from cxQF with our ceqf) LB 0.988328 = TEAM BEST; tsim US 0.991110 / India 0.992145. Vs fin5_frm2: US/India explain +0.000217 of +0.000331 -> France ~+0.00076 over fin5 (~0.967). mx6/tb3 = blQF on US/India (net ~0), so they only differ on France. No blQF/cxQF score files on our side yet.
- 23:31 (user: final ranking = BEST submission, so uploads are free probes) tb3_frm1.5 released (q_tb15 41986): PASS, 5,833,068 links; vs tb3_frm2 France 3,224 rows, US/India 0. Upload order: tb3_frm2 -> mx6_frm2 -> tb3_frm1.5 (-> fin7s_frm1.5 only if a slot is left).
- 23:34 (session a3) decision tweaks on mx7c (tsim): caps +0.000000 (5/5 folds >=0) -> ADOPTED: mx7c_frm2k released (57 rows vs mx7c_frm2: FR 50 IN 1 US 6), PASS. prior0.5 -0.000004 (1/5), caps+prior0.5 -0.000004, prior1 -0.000002 -> rejected (mx7c_frm2q contains prior 0.5: do NOT upload). Also released mx7c_frm1 (fr=-1, France empty 5.76% = US/India 5.77%). Upload order: mx7c_frm2k, mx7c_frm1.5, mx7c_frm1, tb3c_frm2.
- 23:35 (user idea: prioritise ceqf/ceqft on France) new ana/splice.sh: US/India rows from tb3_frm2 + France rows from fin7s -> hy7_frm2 / hy7_frm1.5 (q_hy7 41990, both PASS, candidates identical md5). Diffs check out: vs tb3 only France rows (11,279 / 11,845), vs fin7s only US/India (7,887). Final 4 uploads: hy7_frm2, tb3_frm2, hy7_frm1.5, tb3_frm1.5 (fin7s_frm1.5 dominated by hy7_frm1.5; mx6 dropped).
- 23:48 LB hy7_frm2 = 0.988034 -> implied France ~-0.002 vs blQF, ~-0.0012 vs fin5: ceqf DROP-IN (stage 2 trained on ceq) hurts France; friend's cxQF (trained with ceqf + French rows) is best. Built tb5_frm2 (ana/tb5.py, q_tb5 41992): tb3 with France member fin5s (plain ceq) instead of fin6s; US/India identical to tb3; PASS; vs tb3 2,074 France rows. Remaining uploads: mx6_frm2, tb5_frm2 (then tb3_frm2).
- 23:50 export_for_kushal.tar (fin5 = LB-0.987997 stage 2, raw p, for teammate blend; md5 dd3019ea1d4c189fee0a432283d02975) locally tested with new ana/xtest.sbatch (job 41993, PASS): md5 OK; kcheck rows 8,194,747 / 9,187,520, keys 100%, p bit-identical to er_final (max|dp| 0); shipped oof + shipped dec -> tsim 0.9912756 (US 0.9908797, India 0.9918688), per-fold diff 0; decide.py apply shipped pte + dec + fr=-2 -> matching_results.tsv BYTE-IDENTICAL to versions/fin5_frm2 (5,816,637 links), validator PASS. dec has no fr key (fr=-2 is an apply arg); decide always runs one2one (margin 0.3) before ef.
- 23:55 LB tb5_frm2 = 0.98834 = NEW TEAM BEST (blQF 0.988328): removing the ceqf drop-in member helped France. Built tb6_frm2 (France 0.5 cxQ + 0.25 fin5s + 0.25 fx, US/India = tb5; PASS; vs tb5 1,557 France rows) as the last probe.
- 00:27 (28 Sep) architecture.md written for the team best tb5_frm2 (LB 0.98834): shared pipeline -> 3 stage-2 members (fin5s / cxQ / fx) -> logit team blend (US/India 0.7/0.3/0, France thirds) -> decision (fr=-2, one2one 0.3, ef m=0); reproduction commands for fin3 -> sup3 -> fin5sa -> fin5s -> wblend/tb5 -> qfix apply. Documentation.md still has TBDs and a stale final (fin6c, p1>=0.005).
- 29 Sep 05:30 (session 2c) Final submission folder for the best-local-F0.5 version tb5_frm2 (tsim 0.991528, LB 0.98834): ~/amazon_ml/tb5_submission (not zipped). Base = session ac's cleaned code in TEAM_NAME_submission (fin5_frm2 package, left untouched); added src/team_blend.py (wblend+tb5), src/kcheck.py, run_all.sh steps 18-21 (fin3 -> sup3 -> fin5sa -> fin5s, import+gate teammates' kcxQ/kfx scores from TEAM_SCORES, team blend, decide apply fr=-2); README + Documentation_template updated. Job 42056: package code on er_final member scores rebuilds fin5s (max|dp| 0), blend weights 0.7/0.3/0 (0.991528), and BOTH TSVs byte-identical to versions/tb5_frm2 (md5 5b4ad6c3 / 1cca4a0c); validator PASS. Gap: cxQ/fx training code not in the package (teammates' repos).
