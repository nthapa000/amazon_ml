# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary

The pipeline has five stages:
1. **Normalization.** Names and addresses are cleaned per country. A native-script → Latin token dictionary is learned only from the provided training matches.
2. **Blocking.** Compound keys (name token × rare address word, address word pairs, full name, address codes, name substrings) produce the candidates. A sibling-expansion pass then re-blocks unassigned records against S1 profiles enriched with their confident matches.
3. **Stage-1 matcher.** A LightGBM model scores about 100 hand-crafted similarity and hard-negative features.
4. **Stage-2 matcher.** A second LightGBM adds competition features computed from stage-1 out-of-fold scores, plus a fine-tuned cross-encoder score for uncertain pairs.
5. **Decision.** A one-to-one assignment followed by an expected-F0.5 set selection.

The key idea is to validate on a *test-like* split. We drop 19% of the train Source-1 entities so that orphan records appear at the same rate as in the test set (5.76 Source-2/3 records per Source-1). Every threshold and model choice is made on that split.

---

## 2. Methodology

### 2.1 Problem Analysis

- **Scale.**
  - Train: 2.21M Source-1 (S1), 5.03M Source-2 (S2) and 5.29M Source-3 (S3) records.
  - Test: 1.73M S1, 4.89M S2 and 5.08M S3 records.
  - 5.6% of train S1 entities are singletons. The typical entity has 3–4 matches.
- **One-to-one structure.** Every S2/S3 id links to at most one S1 id (7.64M links, all unique). We enforce this at decision time.
- **Test shift.**
  - Test has 5.76 S2/S3 records per S1, against 4.68 in train, in every country.
  - France (15% of test) never appears in train.
  - The extra records behave like orphans, i.e. records whose owner is absent from S1. We simulate this with the `tsim` split.
- **Name noise.**
  - Indic-script transliterations of English names in Devanagari, Tamil, Telugu, Kannada, Gujarati, Bengali and Gurmukhi (e.g. "राम मार्केटिंग प्राइवेट लिमिटेड" = "Ram Marketing Private Limited").
  - Legal-suffix variation, word transpositions, typos and leet substitutions (Agr0, Live1y).
  - Junk prefixes and suffixes (`--`, `<<`, phone numbers, `(ID: 87671)`).
  - Domain-style names (`maurewilliamscolombier.com`, word order permuted).
  - Invented DBA brand names (Korflux, Evolum, Xylolum).
- **Address noise.**
  - Abbreviations (St/Street, R./Rue, UP/Uttar Pradesh), state names in native script, reordered components.
  - Truncated or padded house numbers (3029→029, 0684→684) and `null` tokens.
  - About 3% of addresses are empty.
- **Hard negatives (from error analysis).** Distractor records copy the address but change a short initials token (GAA→GIA Plastics, BP→TP Piedmont), or keep the name and change the house number.
- **Name reuse.** The generator reuses a small name vocabulary. Many distinct S1 businesses share the same core name ("Royal Builders", "Dental Medicine"), so names alone cannot identify an entity.

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-stage GBDT classifier + cross-encoder re-scoring + metric-aware assignment

**Core Innovation:**
1. **Test-like validation split** (orphan simulation). Thresholds and model choices hold up under the test's higher distractor rate.
2. **Competition features** from stage-1 scores. For each pair: its margin over the S2/S3 record's best alternative S1, and its rank and gap within the S1's candidate list.
3. **Hard-negative features** targeting the generator's distractor patterns.
4. **Learned transliteration** dictionary, mined from training matches only (no external data).
5. **Sibling expansion.** A missed record is reached through an already-matched record of the same business in the other source.

### 2.3 How the key challenges were handled

#### (a) Noisy Source-2 / Source-3 records

**Normalization** (`norm.py`) is applied identically to all three sources before any comparison:
- **Unicode and accents:** NFKC and accent stripping ("Payne Énterprises" → "payne enterprises").
- **Junk removal:** prefixes such as `--`, `<<`, `***`; phone numbers; `(ID: 87671)`; `null` / `<NULL>`; websites attached with `|`.
- **Leet correction:** inside alphabetic tokens only ("Agr0" → "agro", "Live1y" → "lively").
- **Legal forms:** canonicalized (Pvt → private, Ltd → limited, L.L.C. → llc) and stored apart from the *core name*. This makes "LLC Orellana Invsmbens" and "Orellana Investments LLC" comparable on the core name, and it survives word transpositions.
- **Addresses:**
  - Street types, directionals, units and states are canonicalized.
  - House numbers are canonicalized ("AF-0684" → "af-684", "45ND" → "45").
  - Components are compared as sets or component-wise best matches, so reordering ("KANSAS CITY, MO, 630 45ND TERRACE") doesn't matter.
- **Alternate names and domains:** "formerly known as / dba / aka" names are split out and compared separately. Domain names are compared against the concatenated S1 name.

**The matcher tolerates the remaining noise**:
- It uses about 100 fuzzy similarity features.
- It uses **hard-negative features** built from error analysis to separate generator noise from genuinely different businesses:
  - initials-token mismatch at the same address (GAA vs GIA Plastics)
  - house-number *truncation* (3029 vs 029, noise) vs *different number* (113 vs 120, a different business)
- A fine-tuned cross-encoder reads both records jointly for the uncertain pairs.

#### (b) Linking records together

1. **Blocking:** compound keys produce candidates, and the top-K per S1 and per S2/S3 record are kept (Section 3).
2. **Sibling expansion:**
   - Each S1 profile is enriched with the fields of its confident matches (p ≥ 0.9, one-to-one).
   - Still-unassigned S2/S3 records are re-blocked against these enriched profiles.
   - This reaches records whose own text is too far from the S1 record but close to a sibling. It recovers 14% of the pairs blocking missed.
3. **Scoring:**
   - Stage-1 LightGBM scores each pair.
   - Stage 2 adds competition features ("does another S1 claim this record more strongly?", with the margin to the best alternative as the top feature) and cross-encoder scores.
4. **Assignment:**
   - Every S2/S3 record is linked to at most one S1: its best-scoring S1, and only if it beats the runner-up by a margin.
   - The ground truth has exactly this one-to-one structure. Removing the rule costs 0.0015 F0.5 (0.98368 vs 0.98517 for u4).

#### (c) Singletons: identifying businesses with no match

- **The empty list is an explicit decision.**
  - For each S1 the rule compares the expected F0.5 of the best top-k set with P(no match) = Π(1 − p_i).
  - It outputs an empty list whenever that is more likely to score 1.0.
  - The expected-F formula weights precision 2×, the same as the metric.
- **Features teach the model when not to match:**
  - name commonness (many S1 share names like "Dental Medicine", so name similarity alone isn't trusted)
  - whether the S2/S3 name's words occur in S1 names at all
  - initials and house-number conflicts
  - the competition margin
- **Thresholds are tuned under test-like conditions.** On the `tsim` split, 19% of S1 are dropped so their records become orphans, which reproduces the test's 5.76 S2/S3 records per S1. The empty-set decision is therefore tuned where false merges are most likely.
- **Result (validation, u3):**
  - 98,600 of 99,652 true singletons (**98.9%**) are predicted empty.
  - Pair precision is **99.77%**.
  - Singleton errors are about 4% of the remaining loss; about 85% of the remaining loss is missed links, not false merges.

#### (d) Region-specific patterns in names and addresses

- **India:**
  - A native-script → English dictionary is **learned from training matches only**. It covers 1,347 tokens across Devanagari, Tamil, Telugu, Kannada, Gujarati, Bengali and Gurmukhi (e.g. "प्राइवेट लिमिटेड" → "private limited", "ராஜ்" → "raj"). Unseen tokens fall back to romanization.
  - Native-script and abbreviated state names are canonicalized (தமிழ்நாடு / TN → tamil nadu; UP, MH, KA, DL).
  - Indian address vocabulary is normalized: H.No → house, opp → opposite, nr → near, khasra, taluk. So are legacy city names (Bombay → Mumbai, Bangalore → Bengaluru).
  - Unit codes such as "29/101" and "b-1107/1108" are blocking keys. Indian legal forms (Pvt Ltd, LLP, OPC) are handled.
- **US:**
  - State abbreviations ↔ names (NY ↔ New York) and USPS street suffixes (Ave, Blvd, Trl, Ter, Hwy).
  - Directionals (N → north), unit markers (Apt/Ste/Unit/PMB) and US legal forms (LLC, PLLC, P.C., Inc, Corp).
- **France (no training data):**
  - French street types (R./Rue, Av, Bd, All./Allée, Imp, Chem, Fg). "St" means *saint* in France, whereas in the US it means *street*.
  - French legal forms (SARL, SAS, SASU, SCI, EURL, SNC, Cie), handled as legal suffixes.
- **Country is only a blocking filter, never a model feature.** The features are country-agnostic, so the model applies to France without French labels.
- **Test-time drift check for France:**
  - 94% of French S1 have a best-candidate score above 0.9, the same as US and India.
  - 5.1% are predicted empty, against 5.7–5.8% elsewhere.

---

## 3. Candidate Generation (Blocking)

- **Normalization before blocking:**
  - NFKC and unidecode.
  - Learned native-script → Latin token map (1,347 name tokens and 16 address tokens, mined by positional alignment and Dice co-occurrence on train matches).
  - Legal-form canonicalization for US, India and France (pvt→private, ltd→limited, l.l.c.→llc, sarl/sas/sasu/eurl/sci).
  - Street-type and state canonicalization, house-number canonicalization (leading zeros, ordinals).
- **Blocking keys** (always within country; country is a filter only and never a model feature):

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
  | cat / sub | concatenated name prefix; 6-char substrings of single-token names/domains | `okg.com`, permuted domains |

- **Block-size caps and scoring.** Each family has a cap on block size per side. Pairs are scored by the IDF-like key weight Σ 1/√(n1·n2). We keep the top-K1 candidates per S1 and the top-K2 per S2/S3 record.
- **Candidate pairs generated (test):**
  - 78.1M at K1=40 / K2=6.
  - 151.2M at K1=100 / K2=10.
- **Recall on the test-like train split:**

  | version | pair recall | ceiling F0.5 (perfect matcher) | cand/S1 |
  |---|---|---|---|
  | v1: simple keys | 0.921 | 0.955 | 147 |
  | v2: compound keys | 0.953 | 0.979 | 42 |
  | v3: + fw/code/sub/digit-only hn | 0.973 | 0.991 | 42 |
  | v3, K1=100 (test-like split) | 0.978 | 0.993 | 90 |
  | + sibling expansion (test-like split) | 0.981 | 0.9936 | 97 |

  (v1–v3 at K1=40 are on the full train split; the last two rows are on the test-like split.)

- **How we kept true matches from being lost:**
  - Every miss was reviewed by family.
  - Families were added until the remaining misses were mostly generic names with an empty address, which are inherently ambiguous.
  - The top-K is taken per S1 **and** per S2/S3 record, so each record keeps its best S1 candidates.

---

## 4. Matching Model

**Features used** (~100):
- **Name:**
  - rapidfuzz ratio / token_set / token_sort / partial and Jaro-Winkler on the full and core name
  - token Jaccard, IDF-weighted overlap, soft token coverage (JW ≥ 0.88)
  - concatenated-name similarity, domain match, alt-name (fka/dba) match, acronym match
  - legal-form agreement or conflict, script flags
- **Address:**
  - token_set / sort, component-wise best-match similarity, IDF overlap
  - house-number intersection, conflict, subset and first-number equality
  - postal-code match, canonical state match, empty flags
- **Hard-negative:**
  - short initials-token mismatch
  - house-number numeric distance, truncation pattern and length difference
- **Split statistics:**
  - how many S1 share the core name or the address
  - whether the S2/S3 name tokens appear in the S1 vocabulary at all (invented DBA names: 32% match rate vs 0.8%)
  - duplicate-name count among the candidates
- **Blocking evidence:** key score, number of shared keys, key-family bits, rank of the pair among the S1's and the S2/S3 record's candidates.
- **Stage-2 context:** stage-1 score, rank and gap within the S1's candidates; the S2/S3 record's best alternative S1 score and margin; candidate counts and score sums.
- **Cross-encoder:**
  - `cross-encoder/ms-marco-MiniLM-L12-v2` (Apache-2.0, 33M parameters), fine-tuned on "name | address" pairs.
  - Used for the uncertain band (stage-1 p in 0.003–0.997), with 2-fold out-of-fold scores.
  - In-band AUC is 0.972, against 0.971 for stage 1.

**Model type:** LightGBM binary (num_leaves 255, lr 0.08, early stopping). 5-fold GroupKFold by S1 id, which is leak-free because each S2/S3 has at most one owner. Stage 2 is trained on stage-1 out-of-fold predictions.

**Threshold selection method:**
- One-to-one assignment: each S2/S3 record keeps only its best S1, and must beat the runner-up by a margin of 0.2.
- Per S1, pick the top-k set, or the empty set, that maximizes expected F0.5 = 1.25·Σp_top-k / (0.25·Σp + k).
- The method and its parameters are tuned on out-of-fold macro F0.5 on the test-like split.

---

## 5. Results & Error Analysis

| version | change | OOF macro F0.5 |
|---|---|---|
| s1 | blocking v3 + stage-1 LightGBM, threshold 0.7 (train split) | 0.97803 |
| t1 | test-like split + name-statistics features | 0.97961 |
| t2 | + stage-2 competition features | 0.98092 |
| u1 | stage-1 + hard-negative features | 0.98117 |
| u2 | stage-2 + context + cross-encoder L6 | 0.98422 |
| u3 | + cross-encoder L12 wide band | 0.98485 |
| u4 | cross-encoder trained 2 epochs (band AUC 0.972 → 0.977) | 0.98517 |
| u5 | stage 3 (context recomputed from u4) | 0.98517 (no gain, not used) |
| w1 | stage-1 on K100 + sibling-expansion candidates | 0.98203 (stage 1 alone) |
| w2 | stage-2 on the expanded candidates + context + cross-encoders | **0.98634** |

(t1 onwards are measured on the test-like split, which is harder than the plain train split.)

- **F0.5 (macro, validation):** 0.98634 (US 0.98599, India 0.98686; F0.4 0.98880).
- **Decision-rule ablation (u4):**
  - Without one-to-one assignment: 0.98368.
  - With one-to-one plus expected-F: 0.98517.
  - Re-tuning over 48 margin / threshold settings plateaus at 0.98519.
- **Common false positives:**
  - Deliberate hard negatives: same address with changed initials, or same name with a different house number.
  - Generic names with an empty S2/S3 address assigned to the wrong one of several same-name S1s.
  - Invented DBA names at an address shared by two S1 businesses.
- **Common false negatives:**
  - Blocking misses (about half), mainly very common names with an empty or city-only address.
  - Records with empty addresses and heavy name corruption.
  - DBA records with a partial address.

---

## 6. Conclusion

- Most of the gain came from data-driven blocking, error-analysis-driven hard-negative features, and competition-aware stacking.
- The validation split was built to match the test's orphan rate, so decisions tuned offline carry over to the leaderboard.
- The learned transliteration map and country-agnostic features let the model score France without any French labels.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:
- `src/`: `prep.py`, `dicts.py`, `norm.py`, `block.py`, `ext.py`, `feats.py`, `feats2.py`, `xf.py`, `train.py`, `ctx.py`, `ce.py`, `decide.py`, `metric.py`, `mksim.py`
- `run_all.sh`: end-to-end entry point
- `jobs/`: SLURM wrappers
- `README.md`, `requirements.txt`

Entry point: `bash run_all.sh` → `output/matching_results.tsv`, `output/candidate_pairs.tsv`.

**Compliance:**
- No external data, APIs or lookups are used.
- Pretrained models, both fine-tuned only on the provided training data:
  - `cross-encoder/ms-marco-MiniLM-L12-v2` (Apache-2.0, 33M parameters)
  - `cross-encoder/ms-marco-MiniLM-L6-v2` (Apache-2.0, 22M parameters)
- Both are far below the 8B cap.

### B. Additional Results

Blocking family contributions and the ablation log are in `EXPERIMENTS.md`. The version history, with stored submissions, is in `VERSIONS.md`.
