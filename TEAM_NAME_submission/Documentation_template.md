# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** 108 Neurons  
**Team Members:** Mahek Gadiya, Nishant Thapa, Kushal Sharma, Shasank Dixit  
**Submission Date:** 29-09-2026

---

## 1. Executive Summary

We normalize names and addresses with country-aware rules and a transliteration dictionary learned from the training
matches. Candidates come from three passes: compound blocking keys, sibling expansion through already-matched
records, and multilingual-embedding nearest neighbours. A stage-1 LightGBM then scores about 103 first-pass pairs per
Source-1 entity and keeps **5.30 per entity** (p1 ≥ 0.001). These pairs are `candidate_pairs.tsv` and the only pairs
the final model scores.

The final model is a stage-2 LightGBM cascade. It uses competition features, sibling-support features and six
fine-tuned cross-encoders (MiniLM up to a 4B Qwen3 reranker). Links are chosen by a one-to-one assignment and an
expected-F0.5 set rule in which "no match" is an explicit option.

Every decision was validated on a test-like split of the training data. Validation macro F0.5 is **0.991276**, and the
public leaderboard score is **0.987997**.

---

## 2. Methodology

### 2.1 Problem Analysis

- **Scale.**
  - Train: 2,206,821 Source-1 (S1), 5,034,616 Source-2 (S2) and 5,285,603 Source-3 (S3) records.
  - Test: 1,732,544 S1 (India 809,986 · US 663,106 · France 259,452), 4,887,273 S2 and 5,082,316 S3.
  - Comparing all S1 × (S2 ∪ S3) test pairs would mean 17.3 trillion comparisons.
- **Label structure.**
  - 7.64M training links. 5.6% of S1 entities are singletons, and the typical entity has 3–4 matches.
  - Every S2/S3 record links to at most one S1 entity. We enforce this one-to-one structure at decision time.
- **Test shift.**
  - Test has 5.75 S2/S3 records per S1, against 4.68 in train, in every country.
  - The extra records behave like orphans (their business is absent from S1) and are the main source of false merges.
  - France (15% of test S1) never appears in train.
- **Name noise.**
  - Indic-script renderings of English names (Devanagari, Tamil, Telugu, Kannada, Gujarati, Bengali, Gurmukhi).
  - Legal-suffix variation, word transpositions, typos, digit-for-letter substitutions (Agr0, Live1y).
  - Junk tokens (`--`, `<<`, phone numbers, `(ID: 87671)`), domain-style names (`maurewilliamscolombier.com`) and invented trade names.
  - Truncated names ("Green Creative Dynamics" → "Green Creative").
- **Address noise.**
  - Abbreviations (St/Street, R./Rue, UP/Uttar Pradesh), state names in native script, reordered components.
  - Truncated or zero-padded house numbers (3029 → 029, 0684 → 684), `null` tokens.
  - About 3% of addresses are empty.
- **Hard negatives** (found by error analysis of our out-of-fold predictions).
  - Distractors copy an address but change a short initials token (GAA → GIA Plastics).
  - Others keep the name but change the house number (113 vs 120).
  - In France, one generic word is swapped ("Nantes Collège SA" vs "Nantes Amicale SA").
- **Name reuse.** Many distinct S1 businesses share a core name ("Royal Builders"), so a name alone does not identify an entity.
- **France** (inspected on the unlabelled test inputs).
  - S1 writes the region in 86.6% of French addresses, while S2/S3 often write the department ("Bordeaux, Nouvelle-Aquitaine" vs "BORDEAUX, Gironde").
  - House numbers appear as "N°24".
  - Place names are written with or without hyphens (Saint-Nazaire / SAINT NAZAIRE).
  - French ordinals (1er, 3ème), elisions (L'Atelier) and French legal forms (SARL, SAS, EURL, EI…) appear throughout.

### 2.2 Solution Strategy

**Approach Type:** Hybrid. Multi-pass blocking, a two-stage LightGBM cascade with stacked cross-encoder scores, and a
metric-aware assignment.

**Core Innovation:**

1. **Test-like validation split.**
   - 19% of the train S1 entities are dropped, so their S2/S3 records become orphans and the ratio of S2/S3 records to S1 matches test (5.77 vs 5.75).
   - All tuning happens on this split, where false merges occur as they do on test.
   - Seed-to-seed noise of the score is about 0.000004. Our leaderboard results matched the offline scores for US/India.
2. **Competition features.** Stage-1 out-of-fold scores give each pair its margin over the record's best alternative
   S1, and its rank and gap in the S1's candidate list. The margin is the strongest stage-2 feature (49% of the gain).
3. **Collective evidence.**
   - *Sibling expansion* reaches records that share no blocking key with their S1 through an already-matched record of the same business.
   - *Sibling-support features* compare each candidate with the S1's other confident matches, on both raw and normalized text.
4. **Learned candidate filter.** The stage-1 model reduces 103 first-pass pairs per S1 to 5.30. Stage 2 then affords
   100% of the training data and large cross-encoders.
5. **Multilingual cross-encoders on raw text**, stacked as stage-2 features: native scripts, casing and French accents
   stay visible. They moved the French score the most.
6. **Self-training for the unseen country.** A French training set is built only from the provided records: teacher
   pseudo-labels, exact S1–S1 negatives and noised S1 copies.

---

## 3. Candidate Generation (Blocking)

All passes run within one country. No step compares all pairs. Country selects the normalization rules and restricts
blocking to one country; it is never a model feature, and the code treats it as an open set of labels.

- **Blocking keys used:**

  | family | key | purpose |
  |---|---|---|
  | tok / pair | one core-name token; sorted pairs of core-name tokens | typo-robust name match |
  | tokst | name token + state | makes common tokens selective |
  | full | sorted full core name | records without an address |
  | nw | name token × rare address word | common names disambiguated by the street |
  | fw | full name × any address word (incl. city) | very common names with city-only addresses |
  | ww | pairs of rare address words | trade names / renamed businesses at the same address |
  | hw | house number (and its digit-only form) × address word | address match despite noisy names |
  | code | raw address codes (29/101, b-1107/1108) | Indian unit codes |
  | cat / sub | concatenated name prefix; 6-character substrings of single-token names and domains | `okg.com`, run-together names |

  - Each family has a block-size cap per side, which drops overly generic keys.
  - A pair's score is Σ 1/√(n1·n2) over its shared keys, where n1 and n2 are the key's block sizes.
  - We keep the top 100 candidates per S1 **and** the top 10 S1 per S2/S3 record.

- **Sibling expansion:**
  1. Stage-1 model `a1` scores the key candidates.
  2. Each S1 profile is enriched with its confident matches (p ≥ 0.9, one-to-one).
  3. Unassigned S2/S3 records are re-blocked against the enriched profiles, with top-20 per S1 / top-5 per record.
  4. This recovers 13.8% of the true pairs the keys missed.

- **ANN pass:**
  - `intfloat/multilingual-e5-small` (MIT) embeds normalized "core name | address". The model is frozen and used like
    a blocking key; it is never fine-tuned.
  - Exact GPU nearest-neighbour search finds the top 3 S1 for every S2/S3 record within its country.
  - This recovers 43.6% of the true pairs that keys and expansion missed, at 9 new pairs per S1.

- **Learned filter = final candidate set:**
  - Two stage-1 LightGBMs score every first-pass pair: `b1` for key/expansion pairs (80 features) and `b1ann` for ANN
    pairs (81 features, adding embedding similarity and rank).
  - Pairs with p1 ≥ 0.001 form the candidate set. The final matching model scores **only** these pairs. `candidate_pairs.tsv`
    is written from exactly the pairs the final model scored, so every match is a candidate.

- **Candidate pairs generated (test):**

  | stage | pairs | per S1 |
  |---|---|---|
  | compound keys, top-100 / top-10 caps, + sibling expansion + ANN top-3 (first pass) | 178,935,979 | 103.3 |
  | **learned filter p1 ≥ 0.001 = `candidate_pairs.tsv`** | **9,187,520** | **5.30** (France 8.12 · India 4.78 · US 4.83) |

  - The reduction ratio against all 17.3 trillion S1 × (S2 ∪ S3) pairs is 99.99995%.
  - The typical S1 entity has 3.46 true matches, so 5.30 candidates is about 1.5× the true count.
  - French entities keep more candidates because the model is less certain on the country absent from training.

- **How we ensured true matches were not lost.** Recall on the test-like split (share of true pairs present, and the
  F0.5 a perfect matcher would reach on that set):

  | candidate set | pair recall | ceiling F0.5 | candidates / S1 |
  |---|---|---|---|
  | compound keys, top-40 caps | 0.9743 | 0.9917 | 46 |
  | top-100 caps | 0.978 | 0.993 | 90 |
  | + sibling expansion | 0.9808 | 0.9936 | 97 |
  | + ANN top-3 (first pass) | 0.9892 | 0.9967 | 106 |
  | **learned filter p1 ≥ 0.001 (final)** | **0.9890** | | **4.6** |

  - We reviewed missed pairs after every blocking change and added key families until the remaining misses were mostly
    generic names with empty addresses.
  - The top-K cut is taken per S1 and per S2/S3 record, so records of popular names are not crowded out.
  - Two independent passes (expansion, embeddings) reach records that share no key with their S1.
  - The learned filter loses only 0.0002 of pair recall. A fixed top-5 per S1 cap instead would have lost 5 points of
    recall, because entities with many true matches need more candidates.

---

## 4. Matching Model

**Features used** (116 in the final stage 2):

- **Name features:**
  - RapidFuzz ratio, token-set, token-sort and partial ratios, and Jaro-Winkler, on the full and on the core name (legal form removed).
  - Token Jaccard, IDF-weighted overlap, soft token coverage in both directions.
  - Concatenated-name, domain, alternate-name (fka/dba) and acronym matches.
  - Legal-form agreement or conflict; script flags (native-script tokens, tokens unknown to the dictionary).
- **Address features:**
  - Token-set and token-sort ratios, component-wise best match, IDF overlap, soft coverage.
  - House numbers: intersection, conflict, subset, first-number equality.
  - Postal code equality, state equality, empty-address flags.
  - Name tokens that appear inside the other record's address.
- **Other:**
  - **Hard-negative features:**
    - initials-token mismatch;
    - house-number numeric distance, truncation pattern and length difference: truncation is noise, while a different number means a different business.
  - **Split statistics:**
    - how many S1 records share the core name or the address;
    - whether the S2/S3 name's words occur in S1 names at all (invented trade names);
    - duplicate names in the candidate list.
  - **Candidate evidence:** key score, number of shared keys, key-family bits, ranks, expansion and ANN flags, ANN similarity and rank.
  - **Competition features** (from stage-1 scores):
    - stage-1 score, and rank and gap within the S1's candidates;
    - the record's best alternative S1 and the margin to it;
    - candidate counts and score sums.
  - **Sibling support:** name, address, raw-text and house-number similarity between the candidate and the S1's other confident matches.
  - **Cross-encoder scores** (six models, below).

**Cross-encoders.**
- Each model is fine-tuned on "name | address" pairs from the stage-1 uncertain band (0.003 < p1 < 0.997), plus a 15% share of easy pairs.
- Training is 2-fold by S1: the model trained on one half scores the other half (out-of-fold), and both halves score test.
- Their scores enter stage 2 as features, never as the final decision.

| feature | model (licence, size) | text | training | band AUC (stage-1: 0.970) |
|---|---|---|---|---|
| `cev` | cross-encoder/ms-marco-MiniLM-L12-v2 (Apache-2.0, 33M) | normalized | 2 epochs | 0.974* |
| `cem` | cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (Apache-2.0, 118M) | raw | 2 epochs | 0.978 |
| `ceb` | BAAI/bge-reranker-base (MIT, 278M) | raw | 2 epochs | 0.980 |
| `cef` | `ceb` continued on the French self-training set + US/India replay | raw | 1 epoch | 0.981 (US/India) |
| `cer2` | BAAI/bge-reranker-v2-m3 (Apache-2.0, 568M) | raw | 1.6M pairs × 2 epochs per half | 0.984 |
| `ceq` | Qwen/Qwen3-Reranker-4B, LoRA r = 16 on all linear layers + new score head, bf16 (Apache-2.0, 4.0B) | raw | 1.2M pairs × 2 epochs per half | **0.986** |

\* on the band of the first stage-1 model (stage-1 AUC 0.972 there). A seventh model, `cer` (bge-reranker-v2-m3, 600k
pairs), is used only by the teacher that builds the French self-training set.

**Self-training for France.** France has no labels, and the organizers' Q&A allows self-training on the test records
and synthetic pairs built from the provided records. The set has 627,863 pairs and uses no external data:
- 200,000 **pseudo-positives**: the teacher model's French test pairs with p ≥ 0.97 that win the one-to-one assignment
  by a margin ≥ 0.3. The same cut-offs reach 0.9996 precision on the labelled validation split.
- 200,000 **look-alike pseudo-negatives**: p ≤ 0.02, with name or address token-set similarity ≥ 80. Precision 0.9980 on validation.
- 127,863 **exact negatives**: two different S1 records (S1 is deduplicated) that share the address or the core name, with one side noised.
- 100,000 **exact positives**: an S1 record against a noised copy of itself (abbreviations, department vs region, "N°",
  case, accents, legal-form spelling).

**Model type:**
- **LightGBM binary classifiers, 5-fold GroupKFold by S1 id.** This is leak-free because every S2/S3 record has at most
  one owner. Every stacked input is out-of-fold.
- **Stage 1:** 255 leaves.
  - `a1` (key candidates; drives the sibling expansion) and `b1` use learning rate 0.08 on 15% of the S1 entities per fold.
  - `b1ann` uses learning rate 0.05 on all of them.
- **Stage 2 (cascade):**
  - Trained on the p1 ≥ 0.001 pairs of all S1 entities: 8,194,747 pairs on the validation split, 6,117,951 of them positive.
  - Parameters from 30 Optuna trials on validation logloss: 74 leaves, min_data_in_leaf 240, feature_fraction 0.51,
    bagging_fraction 0.88, λ₂ 27.8, min_gain_to_split 0.14. Learning rate 0.05, early stopping.
  - Test scores are the mean of the 5 fold models.
- **Final blend.** Two stage-2 models share everything except one cross-encoder: `fin5a` uses `ceb` and `fin5b` uses
  the French-adapted `cef` (validation 0.991275 each). They are averaged in logit space with weights 0.8 / 0.2, tuned on
  validation F0.5.

**Threshold selection method:** metric-aware decision, tuned on validation macro F0.5.
1. **One-to-one assignment.** Each S2/S3 record goes to its best-scoring S1, and only if that score beats the runner-up
   by a margin of 0.3. A plain 0.5 threshold without this step scores 0.99021 on validation instead of 0.99128.
2. **Expected-F0.5 set selection.** For each S1, choose the top-k set that maximizes
   E[F0.5] ≈ 1.25·Σp_top-k / (0.25·(Σp + m) + k), with m = 0.5. Output an empty list when P(no match) = Π(1 − p_i) is
   larger. Singletons are therefore an explicit decision.
3. **Unseen-country calibration.** For S1 entities whose country does not occur in training, p' = sigmoid(logit p + δ)
   with δ = −2.
   - The code derives this set of countries from the training file and names no country.
   - δ was chosen with two leaderboard comparisons in which only French rows differed: δ = −1 gave +0.000155 and δ = −2 gave +0.000173.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):**
  - Validation: **0.991276** (US 0.990880, India 0.991869; F0.4 0.992924; 5.7% of S1 predicted empty).
  - Public leaderboard: **0.987997**.
  - Test: 5,816,637 links, with 5.9% / 5.8% / 5.8% of France / India / US entities predicted empty.

| version | change | validation F0.5 | public LB |
|---|---|---|---|
| s1 | key blocking + stage-1 LightGBM, threshold 0.7 (plain train split) | 0.97803 | |
| t2 | test-like split + stage-2 competition features | 0.98092 | |
| u2 | + hard-negative features + first cross-encoder | 0.98422 | |
| w2 | + top-100 caps + sibling expansion | 0.98634 | |
| c1a | cascade stage 2 on all S1 entities | 0.98685 | |
| c2 | + sibling support + multilingual raw-text cross-encoder | 0.98776 | 0.982 |
| c6 | + bge-reranker-base | 0.98790 | 0.9835 |
| c5 | + ANN candidates (own stage-1 model, union features) | 0.99077 | 0.98662 |
| fin | consolidated run: + French normalization | 0.99089 | 0.98677 |
| fin, δ = −2 | + unseen-country calibration | 0.99089 | 0.98695 |
| fin3 | + Optuna parameters + bge-reranker-v2-m3 | 0.99112 | |
| **fin5, δ = −2 (final)** | + Qwen3-Reranker-4B + French-adapted `cef`, blend | **0.991276** | **0.987997** |

- **France.**
  - An early probe with every French row emptied located France's share of the leaderboard score.
  - France is about 15% of the score. Its implied F0.5 rose from about 0.955 (c6) to about 0.966 (final), while
    US/India stay close to the validation score.
  - The gain came mostly from the raw-text multilingual cross-encoders and the calibration shift. From `fin` to the
    final model (bge-reranker-v2-m3, French-adapted `cef`, 4B reranker), France rose by about +0.005.
  - France remains the largest gap. The model is less certain there (more pairs with 0.1 < p < 0.9) and slightly under-links.
- **Singletons** (error analysis of an earlier model): 99.4% of true singletons predicted empty; pair precision 99.85%.
  An entity-level "no match" classifier (AUC 0.9993) added only +0.00001, so the expected-F rule already handles them.
- **Common false positives (wrong merges):**
  - Deliberate look-alikes: the same address with changed initials or one swapped generic word, and the same name with a different house number.
  - Address-less records of generic names assigned to the wrong one of several same-name S1 businesses.
  - Invented trade names at an address shared by two S1 businesses.
- **Common false negatives (missed matches):**
  - Records that share no key with their S1 (heavy corruption, truncated names, empty or city-only addresses). Expansion and ANN recover part of them.
  - Truncated names that are genuinely ambiguous, because many businesses share the prefix.
  - Empty-address records of very common names.
- **Tried and not adopted** (no reliable validation gain):
  - a third stage;
  - prefix-ambiguity features;
  - an entity-level gate;
  - XGBoost members;
  - a second expansion round;
  - larger blocking caps;
  - per-country stage-2 models and decision rules;
  - per-source link caps and count priors.

---

## 6. Conclusion

- **Most of the gain came from recall and collective evidence:** keys → sibling expansion → multilingual ANN,
  competition and sibling-support features, and raw-text multilingual cross-encoders stacked into a cascade that trains
  on all the data.
- **The test-like validation split** kept offline decisions aligned with the leaderboard for the countries with labels.
- **For the country absent from training,** leaderboard decomposition located the gap. Raw-text multilingual
  cross-encoders, self-training built only from the provided records, and a single calibration shift reduced it,
  without external data.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` is self-contained:

- `src/`: all source code.
- `README.md`: setup and exact run instructions.
- `requirements.txt`: pinned versions (Python 3.11, torch 2.7.1 + CUDA 12.8).

**Entry points:**
1. `python src/download_models.py`: downloads the six pretrained models at pinned commits. This is the only network step.
2. `ER_DATA=<dataset> bash src/run_all.sh`: runs the 18 steps from the raw TSVs to `output/matching_results.tsv` and
   `output/candidate_pairs.tsv`, then runs the official validator.

**Source files:**

| stage | files |
|---|---|
| normalization | `prep.py`, `norm.py`, `dicts.py` (learned transliteration dictionary) |
| validation split | `mksim.py` |
| blocking | `block.py`, `ext.py`, `expand.py`, `kx.py`, `ann.py`, `annka.py` |
| features | `feats.py`, `xf.py`, `feats2.py`, `ctx.py`, `sup.py` |
| models | `train.py` (LightGBM, cascade), `ce.py` (cross-encoders), `fradapt.py` (self-training set), `tune.py` (Optuna), `blend.py` |
| decision and output | `decide.py`; `metric.py` (the challenge's macro F-beta) |

**Compliance:**
- No external databases, APIs, geocoding, lookups or internet data.
- Normalization uses small hand-written dictionaries (legal forms, street types, US/Indian state codes, French
  department → region names, eight alternate spellings of Indian city names such as Bombay → Mumbai) and a
  transliteration dictionary learned from the training matches. There is no city → region or postal-code table.
- The test records are used without labels, only for blocking statistics and self-training, which the organizers allow.
- Six pretrained models are used, all MIT or Apache-2.0 and at most 4B parameters. They run offline and are fine-tuned
  only on the provided data; the embedding model is used frozen.

### B. Additional Results

- **Blocking families** (share of true pairs each reaches; "only" means no other family reaches them):
  - `hw` 69.6% (only 1.6%)
  - `nw` 77.4%
  - `fw` 60.9%
  - `ww` 55.6%
  - `code` 16.2%
- **Cascade check.** The decision on the cascade subset and on all pairs scores the same, 0.986338 vs 0.986337.
- **Drift check.** Adversarial validation between validation and test pair features gives AUC 0.64 (US) and 0.59 (India).
  The drift is in blocking statistics only.
- **Stage-2 blend members.** `fin5a` 0.991275, `fin5b` 0.991275, blend 0.991276.
- **Fold consistency.** Replacing the 600k-pair bge-reranker-v2-m3 with the full-band one (fin2 → fin3) improved all five
  validation folds, by +0.00015 on average.
