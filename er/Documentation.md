# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary

We normalize names and addresses per country, using a native-script → Latin dictionary learned only from the training matches. We then generate candidates with three complementary passes: compound blocking keys, graph-based sibling expansion, and multilingual-embedding nearest neighbours.

A stage-1 LightGBM then acts as a learned candidate filter. It cuts ~103 first-pass candidates per Source-1 entity to **4.47**, which keeps 98.8% of true pairs and costs 0.00004 F0.5.

The final model scores only those pairs. It is a stage-2 LightGBM with competition features, sibling-support features and three fine-tuned cross-encoders; two of the cross-encoders are multilingual and read the raw text. Links are chosen by a one-to-one assignment and an expected-F0.5 set selection that treats "no match" as an explicit option.

Every choice is validated on a *test-like* split (orphan simulation). Validation macro F0.5 is **0.99087**, and the public leaderboard score is **0.986773**. France, unseen in training, remains the gap: ≈ 0.960 against ≈ 0.991 for US/India.

---

## 2. Methodology

### 2.1 Problem Analysis

- **Scale.**
  - Train: 2.21M Source-1 (S1), 5.03M Source-2 (S2) and 5.29M Source-3 (S3) records.
  - Test: 1.73M S1, 4.89M S2 and 5.08M S3 records.
  - 5.6% of train S1 entities are singletons; the typical entity has 3–4 matches.
- **One-to-one structure.** Every S2/S3 id links to at most one S1 id (7.64M links, all unique). We enforce this at decision time.
- **Test shift.**
  - Test has 5.76 S2/S3 records per S1, against 4.68 in train, in every country.
  - The extra records behave like *orphans*, i.e. records whose owner is absent from S1. They are the main source of false merges.
  - France (15% of test S1) never appears in train.
- **Name noise.**
  - Indic-script transliterations of English names in Devanagari, Tamil, Telugu, Kannada, Gujarati, Bengali and Gurmukhi ("राम मार्केटिंग प्राइवेट लिमिटेड" = "Ram Marketing Private Limited").
  - Legal-suffix variation, word transpositions, typos, leet substitutions (Agr0, Live1y).
  - Junk (`--`, `<<`, phone numbers, `(ID: 87671)`), domain-style names (`maurewilliamscolombier.com`), invented DBA names (Korflux, Evolum).
  - Truncated names ("Green Creative Dynamics" → "Green Creative").
- **Address noise.**
  - Abbreviations (St/Street, R./Rue, UP/Uttar Pradesh), state names in native script, reordered components.
  - Truncated or padded house numbers (3029 → 029, 0684 → 684) and `null` tokens.
  - About 3% of addresses are empty.
- **Hard negatives (from error analysis of our own out-of-fold predictions).** Distractors copy the address but change a short initials token (GAA → GIA Plastics). Others keep the name but change the house number (113 vs 120). In France, one generic word is swapped ("Nantes Collège SA" vs "Nantes Amicale SA").
- **Name reuse.** A small name vocabulary means many distinct S1 businesses share a core name ("Royal Builders", "Dental Medicine"), so names alone do not identify an entity.
- **France-specific formats** (from inspecting the unlabeled test inputs; France has no labels).
  - S1 writes the **region** in 86.6% of French addresses, while S2/S3 write the **department** about 28% of the time ("Bordeaux, Nouvelle-Aquitaine" vs "BORDEAUX, Gironde").
  - House numbers as **"N°24"** (23.7k records).
  - Hyphenated place names with or without hyphens (Saint-Nazaire / SAINT NAZAIRE, ~47% of records).
  - French ordinals (1er, 3ème), elisions (L'Atelier), French legal forms and abbreviations (EI, Ets).

### 2.2 Solution Strategy

**Approach Type:** Hybrid. Multi-pass blocking (keys + graph expansion + ANN), a two-stage GBDT cascade with cross-encoder re-scoring, and metric-aware assignment.

**Core Innovation:**
1. **Test-like validation split.** 19% of train S1 are dropped, so their records become orphans and the S2/S3-per-S1 ratio matches test (5.76). Seed-to-seed noise of the validation score is 0.000004. The leaderboard confirmed the split for US/India (≈ 0.9885 implied vs 0.9879 offline).
2. **Competition features** from stage-1 out-of-fold scores: a pair's margin over the S2/S3 record's best alternative S1, and its rank and gap within the S1's candidate list. The margin is the most important stage-2 feature.
3. **Collective (graph) evidence.** *Sibling expansion* reaches missed records through an already-matched record of the same business. *Sibling-support features* compare each candidate with the S1's other confident matches on raw and normalized text.
4. **Three complementary candidate passes.** Keys, sibling expansion and **multilingual-embedding ANN**; ANN alone recovers 43% of the links the key-based blocking missed.
5. **Multilingual cross-encoders on raw text** (native scripts, casing, French accents) stacked into the GBDT. They also drove most of the French leaderboard gain.
6. **Cascade = learned candidate filter.** Stage 2 is trained on the pairs with stage-1 p ≥ 0.001 (4.6% of first-pass pairs; 0.013% of true pairs lost), which makes 100% of the training data affordable. The final candidate set is stage-1 p ≥ 0.005: 4.47 candidates per S1.
7. **Learned or rule-based normalization only.** The Indic transliteration dictionary is **learned from the training matches**. French normalization is **rule-based** (language and administrative-geography rules, the same kind as the US/Indian state tables) and applies only to French records.

### 2.3 How the key challenges were handled

#### (a) Noisy Source-2 / Source-3 records
- **Normalization** (`norm.py`) is identical for all three sources:
  - NFKC and accent stripping; junk removal; leet correction inside alphabetic tokens.
  - Legal forms canonicalized and stored apart from the *core name* (so "LLC Orellana Invsmbens" ≈ "Orellana Investments LLC").
  - Street types, directionals, units and states canonicalized.
  - House numbers canonicalized (leading zeros, ordinals).
  - Alternate names (fka/dba/aka) and domain names split out and compared separately.
- **The matcher tolerates the remaining noise:**
  - about 115 fuzzy, statistical and graph features
  - hard-negative features: initials mismatch; house-number *truncation* (noise) vs *different number* (a different business)
  - three cross-encoders that read both records jointly

#### (b) Linking records together
1. **Candidates:** compound blocking keys → sibling expansion → ANN neighbours (Section 3).
2. **Stage 1:**
   - LightGBM scores every candidate pair: one model for key/expansion candidates, one for ANN candidates, which have no key evidence.
   - A cascade keeps the pairs with p ≥ 0.001 for stage 2.
3. **Stage 2:** LightGBM adds competition features, sibling-support features and cross-encoder scores.
4. **Assignment:**
   - Every S2/S3 record goes to at most one S1: its best-scoring one, and only if it beats the runner-up by a margin of 0.3.
   - This mirrors the ground truth's one-to-one structure. Removing it costs 0.0015 F0.5.

#### (c) Singletons: identifying businesses with no match
- **The empty list is an explicit decision.** For each S1 the rule compares the expected F0.5 of the best top-k set with P(no match) = Π(1 − p_i), and outputs an empty list whenever that is more likely to score 1.0. The expected-F formula weights precision 2×, like the metric.
- **Features teach the model when not to match:**
  - name commonness
  - whether the S2/S3 name's words occur in S1 names at all
  - initials and house-number conflicts
  - the competition margin
  - disagreement with the S1's other confident matches
- **Tuned under test-like conditions** (orphans present), where false merges are most likely.
- **Checked alternative:** an entity-level "no match" model reached AUC 0.9993 but improved F0.5 by only 0.00001. The rule is already near-optimal.
- **Result (validation, c2 error analysis):**
  - 99,092 of 99,652 true singletons (**99.4%**) predicted empty
  - pair precision **99.85%**
  - singleton errors are ~3% of the remaining loss

#### (d) Region-specific patterns in names and addresses
- **India:**
  - A native-script → English token dictionary (1,347 tokens) is **learned from training matches** by positional alignment and Dice co-occurrence.
  - State names in native script and state abbreviations are canonicalized.
  - Indian address vocabulary is normalized (H.No, opp, nr, khasra).
  - Unit codes (29/101, b-1107/1108) are blocking keys.
- **US:** state abbreviations ↔ names, USPS street suffixes, directionals, unit markers, US legal forms.
- **France** (no training data):
  - French street types (R./Rue, Av, Bd, All./Allée…) and "St" = *saint*.
  - French legal forms (SARL, SAS, SASU, SCI, EURL, SNC, EI, EIRL, GIE…).
  - "N°" house-number prefixes, French ordinals (1er/2e/3ème), hyphen and elision splitting.
  - **Rule-based department → region alignment**, the French counterpart of the US/Indian state tables. S1 addresses end with a region; S2/S3 sometimes with a department (Gironde → Nouvelle-Aquitaine, Nord and Pas-de-Calais → Hauts-de-France, Loire-Atlantique → Pays de la Loire). The region is also used as the address "state".
  - These rules apply only to French records; 0 of 300,000 checked US/India records change.
- **Country** is used only to block within a country, never as a model feature. Unknown countries fall back to generic normalization.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used** (always within a country):

  | family | key | purpose |
  |---|---|---|
  | tok / pair | single core-name token; sorted token pairs | typo-robust name match |
  | tokst | name token + state | common tokens made selective |
  | full | sorted full core name | empty-address records |
  | nw | name token × rare address word | common names disambiguated by street |
  | fw | full name × any address word (incl. city) | very common names with city-only addresses |
  | ww | pairs of rare address words | DBA / renamed businesses at the same address |
  | hw | house number (and digit-only variant) × address word | address match with noisy names |
  | code | raw address codes (29/101, b-1107/1108, ug14) | Indian unit codes |
  | cat / sub | concatenated name prefix; 6-char substrings of single-token names / domains | `okg.com`, permuted domains |

  Each family has a block-size cap per side. Pairs are scored by Σ 1/√(n1·n2) over shared keys. We keep the top 100 candidates per S1 and the top 10 S1 per S2/S3 record.
- **Sibling expansion:**
  - Each S1 profile is enriched with its confident matches (stage-1 p ≥ 0.9, one-to-one).
  - Unassigned S2/S3 records are re-blocked against the enriched profiles.
  - Recovers 14% of the pairs the keys missed.
- **ANN pass:**
  - `intfloat/multilingual-e5-small` (MIT) embeddings of normalized "core name | address". The encoder is used frozen as published (never fine-tuned, no labels), exactly like a blocking key.
  - Exact GPU nearest-neighbour search of S1 for every S2/S3 record within its country; the top 3 neighbours are added.
  - Recovers **43%** of the pairs keys and expansion missed, at 9 new pairs per S1.
- **Learned candidate filter (final candidate set):**
  - The stage-1 LightGBM, trained on the pair features of the first-pass candidates, scores every first-pass pair.
  - Pairs with stage-1 score p1 ≥ **0.005** form the final candidate set. The final matching model (stage-2 LightGBM + cross-encoders + decision rule) runs **only** on these pairs, and every predicted match is one of them. This set is what `candidate_pairs.tsv` contains.
  - The cut was chosen on the test-like split by measuring the final model's F0.5 when matching is restricted to each set:

    | final candidate set | candidates / S1 (test) | France / India / US | pair recall | validation F0.5 | Δ |
    |---|---|---|---|---|---|
    | first pass (keys + expansion + ANN) | 103.3 | | 0.98920 | 0.990894 | |
    | stage-1 p1 ≥ 0.001 | 5.30 | 8.12 / 4.78 / 4.83 | 0.98903 | 0.990894 | 0 |
    | stage-1 p1 ≥ 0.002 | 4.89 | 7.02 / 4.48 / 4.56 | 0.98888 | 0.990889 | −0.000004 |
    | **stage-1 p1 ≥ 0.005 (used)** | **4.47** | 5.97 / 4.18 / 4.23 | **0.98844** | **0.990853** | −0.000041 |
    | stage-1 p1 ≥ 0.01 | 4.21 | 5.39 / 3.98 / 4.02 | 0.98781 | 0.990799 | −0.000095 |
    | stage-1 p1 ≥ 0.02 | 3.99 | 4.93 / 3.81 / 3.85 | 0.98689 | 0.990714 | −0.000180 |
    | top-5 per S1 (fixed cap) | 4.12 | 4.54 / 4.04 / 4.05 | 0.94180 | 0.985322 | −0.005572 |

  - A score cut beats a fixed per-S1 cap: entities with many true matches keep all of them.
  - The floor is close. S1 entities have 3.46 true matches on average, so 4.47 candidates is 1.3× the true count.
  - French entities keep more candidates (5.97) because the model is less certain on the unseen country.
- **Candidate pairs (test):**
  - key blocking, top-40 caps: 78.1M
  - top-100 caps: 151.2M
  - \+ sibling expansion: 161.7M
  - \+ ANN: 176.7M (first pass, 103 per S1)
  - **learned filter p1 ≥ 0.005: 7.74M = 4.47 per S1**, the content of `candidate_pairs.tsv`
  - reduction ratio 99.99996% vs the 17.3 trillion possible S1 × (S2 ∪ S3) pairs
- **Scalability.** No step compares all pairs.
  - Blocking runs within a country, with block-size caps per key family and fixed top-K (100 per S1, 10 per S2/S3 record). Its cost grows with the number of records, not their square.
  - The ANN pass is a fixed top-3 nearest-neighbour search within a country. Exact search is used here; the same step runs on a standard approximate vector index at larger scale.
  - The learned filter is one GBDT pass, linear in the first-pass candidates.
- **Recall on the test-like split:**

  | candidate set | pair recall | ceiling F0.5 (perfect matcher) | candidates / S1 |
  |---|---|---|---|
  | simple keys (v1) | 0.921 | 0.955 | 147 |
  | compound keys, top-40 | 0.974 | 0.992 | 46 |
  | top-100 caps | 0.978 | 0.993 | 90 |
  | + sibling expansion | 0.981 | 0.9936 | 97 |
  | + ANN top-3 (first pass) | 0.989 | 0.9967 | 106 |
  | **learned filter p1 ≥ 0.005 (final)** | **0.988** | **0.9965** | **4.1** |

  The first row is on the full train split.
- **How we ensured true matches were not lost:**
  - Missed pairs were reviewed after every blocking change, and key families were added until the remaining misses were mostly generic names with an empty address.
  - Top-K is taken per S1 **and** per S2/S3 record.
  - Two independent passes (graph expansion, embeddings) reach records that share no key with their S1.

---

## 4. Matching Model

**Features used** (~115):
- **Name:**
  - rapidfuzz ratio / token_set / token_sort / partial and Jaro-Winkler on the full and core name
  - token Jaccard, IDF-weighted overlap, soft token coverage
  - concatenated-name, domain, alt-name and acronym matches
  - legal-form agreement or conflict, script flags
- **Address:**
  - token_set / sort, component-wise best match, IDF overlap
  - house-number intersection, conflict, subset, first-number equality
  - postal code, state match, empty flags
- **Hard-negative:** initials-token mismatch; house-number numeric distance, truncation pattern and length difference.
- **Split statistics:**
  - how many S1 share the core name or the address
  - whether the S2/S3 name's words appear in the S1 vocabulary (invented DBA names: 32% match rate vs 0.8%)
  - duplicate names among the candidates
- **Candidate evidence:** key score, number of shared keys, key-family bits, candidate ranks, expansion and ANN flags, ANN similarity and rank.
- **Stage-2 context (competition):**
  - stage-1 score, rank and gap within the S1's candidates
  - the S2/S3 record's best alternative S1 and the margin to it
  - candidate counts and score sums
- **Sibling support:** similarity (normalized and raw text, name and address, house number) between the candidate and the S1's other confident matches.
- **Cross-encoders.** Fine-tuned on "name | address" pairs of the uncertain stage-1 band (p in 0.003–0.997), 2-fold out-of-fold by S1:

  | model | licence, params | input | in-band AUC (stage 1: 0.971) |
  |---|---|---|---|
  | `cross-encoder/ms-marco-MiniLM-L12-v2` | Apache-2.0, 33M | normalized text | 0.973 |
  | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Apache-2.0, 118M | raw text (multilingual) | 0.979 |
  | `BAAI/bge-reranker-base` | MIT, 278M | raw text (multilingual) | 0.980 |

**Model type:**
- LightGBM binary, 5-fold GroupKFold by S1 id. This is leak-free because each S2/S3 has at most one owner, and every stacked input is out-of-fold.
- **Stage 1:** num_leaves 255, lr 0.08, trained on 15% of S1 per fold, scoring all candidates. A separate stage-1 model handles ANN candidates.
- **Stage 2 (cascade):** only pairs with stage-1 p ≥ 0.001, trained on 100% of S1, lr 0.05.

**Threshold selection method:**
- One-to-one assignment with margin 0.3.
- Then, per S1, the top-k set or the empty set that maximizes expected F0.5 = 1.25·Σp_top-k / (0.25·(Σp + m) + k).
- Parameters are tuned on out-of-fold macro F0.5 on the test-like split. Re-tuning over 48 settings moved the score by ±0.00002, so this is not a source of overfitting.
- For France only, which has no labels, a single calibration shift of the French pair probabilities is chosen with France-only leaderboard comparisons. US/India rows stay identical in those comparisons, so the difference measures France alone.

---

## 5. Results & Error Analysis

| version | change | validation F0.5 (test-like split) | public LB |
|---|---|---|---|
| s1 | blocking + stage-1 LightGBM, threshold 0.7 (plain train split) | 0.97803 | |
| t2 | test-like split + stage-2 competition features | 0.98092 | |
| u2 | + hard-negative features + cross-encoder | 0.98422 | |
| u4 | + larger cross-encoder, 2 epochs | 0.98517 | |
| w2 | + top-100 caps + sibling expansion | 0.98634 | |
| w2a | stage 2 on 50% of S1 | 0.98670 | |
| c1a | cascade, 100% of S1 | 0.98685 | |
| c2 | + sibling support + multilingual cross-encoder (raw text) | 0.98776 | 0.982 |
| c6 | + bge-reranker-base (raw text) | 0.98790 | 0.9835 |
| c5 | c2 + **ANN candidates** (own stage 1 + union features) | 0.99077 | 0.986624 |
| fin | consolidated end-to-end run from raw TSVs: c5 + bge + French normalization | 0.99089 | 0.986773 |
| fin2b | fin + Optuna stage-2 parameters (30 trials) | 0.99091 | |
| finc | fin, final candidate set = learned filter p1 ≥ 0.005 (103 → 4.47 per S1) | 0.99085 | |
| **fin2bc** | fin2b + final candidate set p1 ≥ 0.005 | **0.99087** | |

- **Not adopted** (gain below the 0.0001 acceptance bar):
  - stage 3 (+0.00000)
  - prefix-ambiguity features (+0.00007)
  - cross-encoder retrained on another band (+0.00006)
  - entity-level gate (+0.00001)
  - XGBoost member and blend (+0.00003)
  - second expansion round (0.6% recall)
  - larger caps (ceiling +0.0004)
- **Optuna on stage 2 (adopted after an unbiased check).**
  - 30 trials all prefer smaller, more regularized trees: logloss −0.8%.
  - 5-fold F0.5 gain +0.000014 is below our bar, so we checked selection bias directly. On folds 1–4, which the search never saw, the gain is +0.000020; on fold 0 it is −0.000010.
  - The gain is small but not an artefact of tuning, so the tuned parameters are kept.
- **F_0.5 score (macro, validation), final candidate set:** 0.99087 (US 0.99045, India 0.99149; F0.4 0.99256).
- **France on the leaderboard.**
  - fin vs c5: +0.00015 on the public leaderboard. The US/India validation gain explains ≈ +0.0001 of it, so France moved only ≈ 0.9595 → 0.9598.
  - The rule-based French normalization changed 9.9% of French predictions, but its net effect was about neutral.
  - The model is 3–4× less certain on French pairs (0.41 pairs per S1 with 0.1 < p < 0.9, against 0.11–0.14 for US/India on test and validation). This motivates the France-only calibration shift (Section 4) and the multilingual cross-encoders.
- **Leaderboard decomposition.** c6 scored 0.9835; the same file with every French row emptied scored 0.849. With France weight 0.1497 this implies:
  - France ≈ 0.955
  - US + India ≈ 0.9885, consistent with validation
  - The raw-text multilingual cross-encoder raised France from ≈ 0.946 (c2) to ≈ 0.955 (c6). The French normalization fixes target the rest of that gap.
- **Common false positives (wrong merges):**
  - Deliberate distractors: the same address with changed initials or one swapped generic word (France), or the same name with a different house number.
  - Address-less records of generic names assigned to the wrong one of several same-name S1 businesses.
  - Invented DBA names at an address shared by two S1 businesses.
- **Common false negatives (missed matches):**
  - Records that share no key with their S1: heavy corruption, truncated names, empty or city-only addresses. The sibling and ANN passes now recover part of them.
  - Truncated names that are genuinely ambiguous (many businesses share the prefix).
  - Empty-address records of very common names.

---

## 6. Conclusion

- **The biggest gains came from recall:**
  - key blocking → sibling expansion → multilingual ANN
  - collective evidence (competition and sibling features)
  - raw-text multilingual cross-encoders, stacked into a cascade that can train on all the data
- **A test-like validation split** kept offline decisions aligned with the leaderboard for the countries we had labels for.
- **For the unseen country, France,** leaderboard decomposition located the gap. It was closed with rule-based French normalization, never with external data.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:

- **Normalization and blocking:**
  - `src/prep.py`: TSV → parquet; parallel normalization
  - `src/dicts.py`: learns the transliteration dictionary from the train matches
  - `src/norm.py`: name and address normalization
  - `src/mksim.py`: test-like split
  - `src/block.py`: compound-key blocking and recall report
  - `src/ext.py`: top-100 caps
  - `src/expand.py`, `src/kx.py`: sibling expansion
  - `src/ann.py`, `src/annka.py`: ANN candidates and union
- **Features and models:**
  - `src/feats.py`, `src/feats2.py`, `src/xf.py`: pair features
  - `src/ctx.py`: competition features
  - `src/sup.py`: sibling support
  - `src/ce.py`: cross-encoders, which save their weights
  - `src/train.py`: LightGBM, cascade option `sub=`
  - `src/cands.py`: final candidate set (stage-1 learned filter p1 ≥ 0.005), restricts final predictions to it
  - `src/decide.py`: one-to-one assignment, expected-F selection, France calibration, output writing
  - `src/metric.py`: the exact README metric
  - `src/blend.py`, `src/gate.py`, `src/tune.py`, `src/snap.py`: experiment utilities
- **Entry point:**
  - `jobs/final.dag` runs the whole pipeline from the raw TSVs on SLURM via `python3 jobs/dag.py jobs/final.dag` (11 dependent jobs).
  - `run_all.sh` lists the same steps sequentially.
  - Outputs: `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
- **Environment:** `requirements.txt` (Python 3.11, polars, rapidfuzz, lightgbm, torch, transformers). Hardware: CPU nodes with 32 cores / 240 GB RAM and one NVIDIA A30 per GPU step.

**Compliance:**
- **No external data, APIs, geocoding or lookups.**
- **Normalization is rule-based or learned from the training data only:**
  - rules: abbreviations, legal forms, ordinals, US/Indian state codes, French department → region table
  - learned: the Indic transliteration dictionary (from training matches)
- **Unlabeled test inputs are used only as inference inputs:** IDF and name-frequency statistics, competition features, sibling expansion and ANN over test records. **No test labels, and no training of model weights on test inputs.**
- **Pretrained models, all fine-tuned only on the provided training data and far below the 8B cap:**
  - MiniLM-L12 (Apache-2.0, 33M)
  - mMiniLMv2-L12 (Apache-2.0, 118M)
  - bge-reranker-base (MIT, 278M)
  - multilingual-e5-small (MIT, 118M, embeddings for ANN)
  - Intermediate versions also used MiniLM-L6 (Apache-2.0, 22M).
- **LightGBM** is MIT-licensed. The text-normalization library Unidecode is GPL (a library, not a model).
- **Reproducibility:** deterministic seeds. Results are reproducible up to tie-breaking order in joins. Re-normalizing 300k French records with the packaged code reproduces the final run exactly.

### B. Additional Results

- **Blocking family contributions** (share of true pairs reached; the "only" figure is reached by no other family):
  - `hw`: 69.6% (only 1.6%)
  - `nw`: 77.4%
  - `ww`: 55.6%
  - `fw`: 60.9%
  - `code`: 16.2%
- **Cascade parity:** decision on the cascade subset vs all pairs is 0.986338 vs 0.986337.
- **Drift audit** (adversarial validation, pair features, test-like split vs test): US AUC 0.64, India 0.59. The drift is in blocking statistics only.
- The full ablation log is in `EXPERIMENTS.md`, and the version history with stored submissions in `VERSIONS.md`.
