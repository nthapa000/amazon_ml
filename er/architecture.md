# Architecture: team solution `tb5` (public LB 0.98834)

This document describes the system behind our best leaderboard submission, `tb5_frm2`. It is a three-member team
blend built on one shared candidate pipeline:

- **Shared by all members:** normalisation, candidate generation and a stage-1 LightGBM filter. They produce one
  candidate set of 9.19M test pairs.
- **Three independent stage-2 matchers** score exactly those pairs.
- **The team blend** averages the three scores in logit space. US and India use weights tuned on validation data;
  France has no labels, so it gets equal weights.
- **One decision rule** turns the blended scores into match lists: a France calibration shift, a one-to-one
  assignment and an expected-F0.5 set selection that can output an empty list.

| | |
|---|---|
| Public leaderboard, macro F0.5 | **0.98834** (team best) |
| Validation macro F0.5 (test-like split, US + India) | 0.991528 (US 0.991122, India 0.992135; F0.4 0.993130) |
| Test S1 entities | 1,732,544 (India 809,986 · US 663,106 · France 259,452) |
| Candidate pairs submitted (`candidate_pairs.tsv`) | 9,187,520 = 5.30 per S1 |
| Predicted links (`matching_results.tsv`) | 5,828,829; 5.8% of S1 predicted empty in each country |

**Terms used below**

| term | meaning |
|---|---|
| S1, S2, S3 | the three sources; every S2/S3 record belongs to at most one S1 entity |
| tsim | test-like validation split built from train (Section 2) |
| p1 | stage-1 match probability of a candidate pair |
| `fin5s` | our stage-2 member |
| `cxQ` | stage-2 member of teammate A (m25csa011) |
| `fx` | stage-2 member of teammate B (Kushal) |
| `tb5` | the team blend of the three members; `tb5_frm2` = tb5 with France shift δ = −2 |

---

## 1. System overview

```
 raw TSVs: S1, S2, S3 for train and test
        │
        ▼
 [1] normalisation (norm.py, dicts.py) + test-like split tsim (mksim.py)
        │
        ▼
 [2] candidate generation, always within one country
     compound blocking keys ─► sibling expansion ─► multilingual-e5 ANN (top 3)
     test: 176.7M pairs = 103 per S1; pair recall 0.989 (tsim)
        │
        ▼
 [3] stage 1: two LightGBMs (key/expansion pairs, ANN pairs) ─► p1
     cascade: keep p1 ≥ 0.001 ─► 9,187,520 test pairs (5.30 per S1)
        │                         = candidate_pairs.tsv, identical for every member
        │
        ├───────────────────────────┬────────────────────────────┐
        ▼                           ▼                            ▼
 [4a] fin5s (ours)            [4b] cxQ (teammate A)        [4c] fx (teammate B)
  2 LightGBMs, logit-blended   LightGBM                     LightGBM
  6 cross-encoders incl.       bge-v2-m3 CEs + our Qwen     bge-v2-m3 CEs (one French-
  Qwen3-Reranker-4B            + 1.66M French pseudo-       adapted) + city→region
                               labelled rows                geo features
        └─────────────┬─────────────┴────────────────────────────┘
                      ▼
 [5] team blend, logit space
     US, India : 0.7 cxQ + 0.3 fin5s + 0.0 fx   (simplex search on tsim OOF)
     France    : 1/3 each                       (no labels: equal weights)
                      │
                      ▼
 [6] decision
     France logit shift δ = −2 ─► one-to-one assignment (margin 0.3)
     ─► per S1: expected-F0.5-optimal top-k set, or the empty set
                      │
                      ▼
     matching_results.tsv + candidate_pairs.tsv
```

---

## 2. Validation design

- **Test-like split (`tsim`, `src/mksim.py`).**
  - 19% of train S1 entities are dropped. Their S2/S3 records become *orphans*, i.e. records whose owner is not in S1.
  - This matches the test ratio of 5.76 S2/S3 records per S1, against 4.68 in raw train.
  - Orphans are the main source of false merges on test, so decisions are tuned where they occur.
- **Folds.** 5-fold GroupKFold by S1 id.
  - This is leak-free because each S2/S3 record has at most one owner.
  - Every stacked input is out-of-fold on tsim: stage-1 scores, cross-encoder scores and sibling support.
- **Scope.**
  - tsim contains only US and India. France is absent from train, so it can only be measured on the leaderboard,
    through France-only comparisons where the US/India rows are identical.
  - Every member's tsim scores were checked before blending with `ana/kcheck.py`. Checks: identical pair keys (100%
    overlap), identical labels, exact row counts (8,194,747 tsim, 9,187,520 test).

## 3. Shared pipeline: normalisation, candidates, stage 1

All three members consume the same output of this part: the same pairs, with the same `block.enc` int64 ids.

1. **Normalisation** (`src/prep.py`, `src/dicts.py`, `src/norm.py`).
   - The same rules apply to all three sources: NFKC and accent folding, junk removal, legal forms split off the core
     name, and canonical street types, units, states and house numbers.
   - The Indic-script → Latin dictionary (1,347 tokens) is **learned from the training matches**.
   - French rules (street types, legal forms, "N°" numbers, department → region) apply to French records only.
2. **Candidate generation.** No step compares all pairs, and every step runs within one country.
   - **Compound blocking keys** (`src/block.py`, `src/ext.py`): key families such as name token × rare address word,
     house number × address word, and full name. There are 11 families, each with block-size caps. We keep the top 100 candidates per
     S1 and the top 10 S1 per S2/S3 record.
   - **Sibling expansion** (`src/expand.py`, `src/kx.py`): each S1 profile is enriched with its confident stage-1
     matches (p ≥ 0.9, one-to-one). Unassigned S2/S3 records are then re-blocked against it.
   - **ANN** (`src/ann.py`, `src/annka.py`): frozen `intfloat/multilingual-e5-small` embeddings of
     "core name | address", exact top-3 neighbours within the country. This recovers 43% of the pairs that keys and
     expansion missed.
3. **Stage 1** (`src/train.py`).
   - Two LightGBMs over about 115 pair features: `b1` for key/expansion pairs, trained on 15% of S1 per fold, and
     `b1ann` for ANN pairs. Their union is `b1ka`.
   - **Cascade:** only pairs with p1 ≥ 0.001 go to stage 2. That is 4.6% of first-pass pairs, losing 0.013% of true
     pairs, so stage 2 can train on 100% of S1.
   - This set is also the submitted candidate set:

     | set | test pairs | per S1 (France / India / US) |
     |---|---|---|
     | first pass | 176.7M | 103 |
     | p1 ≥ 0.001 (used) | 9,187,520 | 5.30 (8.12 / 4.78 / 4.83) |

## 4. Cross-encoders

Cross-encoders are fine-tuned on "name | address" pairs from the stage-1 uncertain band (p1 in 0.003–0.997). They are
trained 2-fold by S1 on tsim, so the scores fed to stage 2 are out-of-fold. Their scores enter stage 2 as features,
never as the final decision.

| column | model (licence, size) | text | used by | band AUC |
|---|---|---|---|---|
| `cev` | `cross-encoder/ms-marco-MiniLM-L12-v2` (Apache-2.0, 33M) | normalised | fin5s, fx | 0.973 |
| `cem` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (Apache-2.0, 118M) | raw | fin5s, fx | 0.979 |
| `ceb` | `BAAI/bge-reranker-base` (MIT, 278M) | raw | fin5s, fx | 0.980 |
| `cef` | `ceb` continued on the French self-training set + US/India replay | raw | fin5s | 0.981 (US/India) |
| `cer2` | `BAAI/bge-reranker-v2-m3` (Apache-2.0, 568M) | raw | fin5s | 0.984 |
| `ceq` | `Qwen/Qwen3-Reranker-4B`, LoRA r = 16 + score head, bf16 (Apache-2.0, 4.0B) | raw | fin5s, cxQ | **0.986** |
| `cx`, `cxs` | bge-reranker-v2-m3, teammate A's recipe; `cxs` adds 20% synthetic train pairs | raw | cxQ (`cx` also fx) | |
| `cfr` | bge-reranker-v2-m3 French CE (scores French test pairs only) | raw | cxQ | |
| `cxf` | bge-reranker-v2-m3 + French synthetic and pseudo-labelled pairs | raw | fx | 0.984 |

The **French self-training set** (`src/fradapt.py`) exists because France has no labels. It is built only from the
provided records:
- teacher pseudo-labels on French test pairs: positives with p ≥ 0.97 that win the one-to-one assignment by ≥ 0.3,
  and look-alike negatives with p ≤ 0.02
- exact negatives: two different S1 records that share an address or core name
- exact positives: an S1 record against a noised copy of itself

## 5. Stage-2 members

All three are LightGBM binary classifiers trained on the tsim cascade set (8,194,747 pairs), scoring the same 9,187,520
test pairs.

| member | inputs beyond the shared pair features | tsim F0.5 (US / India) | LB alone |
|---|---|---|---|
| **fin5s** (ours) | logit blend 0.8 `fin5sa` + 0.2 `fin5b` (weight tuned on OOF). Both have Optuna parameters (30 trials), lr 0.05, 4,000 rounds, and use competition features (`ctx`), `cev`, `cem`, `cer2`, `ceq`. `fin5sa` adds `ceb` and sibling support recomputed from an earlier stage-2 model (`sup3`); `fin5b` adds `cef` and stage-1 sibling support (`sup`) | 0.991298 (0.99090 / 0.99189) | not uploaded alone; predecessor `fin5` 0.987997 |
| **cxQ** (teammate A) | `cx`, `cxs`, our `ceq`, `cfr`; extra pair features (`xf3`), competition features (`ctx2x`, stage-3 `c3`) and geo features; **1,660,380 French pseudo-labelled test pairs** appended as training rows (never in validation); lr 0.02, 8,000 rounds | 0.991508 (0.991095 / 0.992127) | not uploaded alone |
| **fx** (teammate B) | `ceb`, `cem`, `cev`, `cx`, `cxf`; geo features (missing state filled from the city; city → region learned from the provided records); lr 0.05, 4,000 rounds | 0.99112 | 0.987862 (with its own one-to-one normalisation, not used in tb5) |

- **Competition features** (`src/ctx.py`) compare a pair with the alternatives, using out-of-fold stage-1 scores:
  - the pair's margin over the S2/S3 record's best alternative S1; this is the strongest stage-2 feature
  - its rank and gap within the S1's candidates
- **Sibling support** (`src/sup.py`) compares a candidate with the S1's other confident matches, on raw and
  normalised name, address and house number.
- **Complementarity.** The members' scores are highly correlated (0.9986 between cxQ and fin5s), but their errors
  differ:
  - teammate A's model is stronger on US/India (+0.00021 tsim)
  - ours is stronger on France (about +0.0026 France F, from leaderboard decomposition)

## 6. Team blend (`ana/wblend.py`, `ana/tb5.py`)

- **Averaging.** For each pair, `z = Σ w_m · logit(clip(p_m, 1e-6, 1 − 1e-6)) / Σ w_m`, and then `p = sigmoid(z)`. All
  members score every pair, so no member is ever missing.
- **Weights for US and India (seen countries).** A simplex grid search (step 0.25, then ±0.05 refinement) maximises
  tsim OOF macro F0.5 under the fixed decision rule of Section 7. Result:
  - cxQ **0.7**, fin5s **0.3**, fx **0.0**
  - 0.991528, which beats the best single member (cxQ) by +0.000020 on 5/5 folds and fin5s by +0.000230 on 5/5 folds
  - fx adds nothing on tsim
- **Weights for France (unseen countries).** Weights tuned on US/India say nothing about a country without labels, so
  France uses **equal thirds**. The set of unseen countries is computed from the training file; no country is named in
  the code.
- **Why fin5s rather than fin6s in the France blend.**
  - `fin6s` is fin5s with French-adapted `ceqf` scores swapped into the test features at predict time.
  - The leaderboard showed that this drop-in hurts France. Its stage 2 was trained on `ceq` scores, not `ceqf`.
  - `tb5` therefore uses fin5s, whose train and test features match. It became the team best at 0.98834, ahead of
    teammate A's own blend at 0.988328.

## 7. Decision layer (`src/decide.py`; applied with `ana/qfix.py apply`)

The rule is applied in order to the blended score p:

1. **Unseen-country calibration.** For France pairs, `p' = sigmoid(logit p + δ)` with **δ = −2**.
   - δ was chosen with France-only leaderboard probes on an earlier model: δ = −1 gave +0.000155 and δ = −2 gave
     +0.000173. About 23% of the extra links removed at −2 were wrong, which is near break-even, so we stopped there.
   - At δ = −2, France's predicted-empty rate (5.85%) is close to US and India (5.77%).
2. **One-to-one assignment.** Each S2/S3 record keeps only its best S1. It is dropped unless it beats its runner-up by
   **≥ 0.3**, or it has a single candidate. This mirrors the ground truth, where every S2/S3 id links to at most one S1.
3. **Expected-F0.5 set selection.** For each S1:
   - Keep candidates with q = p ≥ 0.02 and sort them by q.
   - Pick the k that maximises `1.25 · C_k / (0.25 · S + k)`, where C_k is the sum of the top-k q and S is the sum
     over all candidates. Parameters: m = 0, a = 1.
   - Output that set only if its expected F0.5 exceeds P(no match) = Π(1 − q). Otherwise output an **empty list**,
     which scores 1.0 when the S1 truly has no match.
4. **Writing the outputs.** `matching_results.tsv` lists one row per test S1 (1,732,544 rows). `candidate_pairs.tsv`
   holds the 9,187,520 cascade pairs; every predicted link is among them.

The rule is fixed, the same one used to tune the blend weights. Re-tuning it over 48 settings moved single-model scores
by only ±0.00002.

## 8. Leaderboard lineage

| submission | what changed | tsim F0.5 | public LB |
|---|---|---|---|
| `fin5_frm2` | our stage 2 alone (Qwen 4B cross-encoder added) | 0.991276 | 0.987997 |
| `fx` | teammate B alone | 0.99112 | 0.987862 |
| `bl1` | teammate A, earlier blend | 0.991336 | 0.987655 |
| `hy7_frm2` | US/India from the team blend, France from our `fin7s` (ceqf drop-in) | 0.991528 | 0.988034 |
| `blQF` | teammate A's final blend (cxQF + two cxQ seeds) | (US 0.991110, India 0.992145) | 0.988328 |
| **`tb5_frm2`** | **team blend of this document** | **0.991528** | **0.98834** |

A last probe, `tb6_frm2`, uses France weights 0.5 cxQ / 0.25 fin5s / 0.25 fx, with US/India as in tb5. Its score is not
recorded in our logs. Private-leaderboard scores are only revealed after the challenge.

## 9. Design decisions

| decision | why |
|---|---|
| One shared cascade set for all members | Blending becomes a per-pair average with no missing scores, and the candidate file is the same whichever member wins |
| Learned stage-1 filter instead of a fixed top-k | 5.30 candidates per S1 keep 98.9% of true pairs. A fixed top-5 would cost −0.0056 F0.5, because entities with many matches lose some |
| Un-pruned set (p1 ≥ 0.001) for the best submission | A pruned 4.47-per-S1 variant (p1 ≥ 0.005, `tb3c_frm2`) costs −0.00005 tsim. Kept as the smaller-candidate-set option |
| Logit-space averaging | Members are LightGBMs trained on the same labels, so their scales are comparable; averaging logits keeps confident votes |
| Separate weights for unseen countries | tsim cannot rank members on France. Equal weights hedge between our France strength and teammate A's US/India strength |
| Calibration shift only for unseen countries | The model is 3–4× less certain on French pairs; the shift is the one France-specific knob, set by France-only leaderboard probes |
| One-to-one assignment before set selection | Matches the data's structure; removing it costs 0.0015 F0.5 |
| An explicit empty-set option | Singletons score 1.0 only when predicted empty; 99.4% of validation singletons were predicted empty |

## 10. Reproducing `tb5_frm2`

**Environment:** `jobs/env.sh` sets `ER_DATA`, `ER_WORK`, `HF_HOME` and `PY`. Release outputs go to `ER_OUT`
(default `/scratch/m25csa019/er_out`).
- CPU steps: 32 cores and about 230 GB RAM.
- GPU steps: one CUDA GPU (A30 class). The 4B reranker needs a large-memory GPU; it was run on a 97 GB RTX PRO 6000.
- On SLURM, `python3 jobs/dag.py jobs/final.dag` runs the same steps as `run_all.sh`.

1. **Shared pipeline, cross-encoders and `fin5a` / `fin5b`:** steps 1–19 of `run_all.sh`.
2. **Our member `fin5s`.** An earlier stage 2 (`fin3`) → sibling support `sup3` from its scores → `fin5sa` → blend:
   ```bash
   $PY src/train.py split=tsim tr=f_cand_tsim_ka te=f_cand_test_ka sub=oof_b1ka:pte_b1ka:0.001 frac=1.0 lr=0.05 seed=1 rounds=4000 extra=xf_tsim_ka+f2_tsim_ka+sib_tsim_ka+annf_tsim_ka+ctx_b1ka_train+cev_a1_train+sup_ka_tsim+cem_b1ka_train+ceb_b1ka_train+cer2_b1ka_train:xf_test_ka+f2_test_ka+sib_test_ka+annf_test_ka+ctx_b1ka_test+cev_a1_test+sup_ka_test+cem_b1ka_test+ceb_b1ka_test+cer2_b1ka_test tag=fin3 pfile=params_fin.json
   $PY src/sup.py split=tsim pred=oof_fin3 thr=0 sib_thr=0.9 out=sup3_ka_tsim
   $PY src/sup.py split=test pred=pte_fin3 thr=0 sib_thr=0.9 out=sup3_ka_test
   $PY src/train.py split=tsim tr=f_cand_tsim_ka te=f_cand_test_ka sub=oof_b1ka:pte_b1ka:0.001 frac=1.0 lr=0.05 seed=1 rounds=4000 pfile=params_fin.json extra=xf_tsim_ka+f2_tsim_ka+sib_tsim_ka+annf_tsim_ka+ctx_b1ka_train+cev_a1_train+sup3_ka_tsim+cem_b1ka_train+ceb_b1ka_train+cer2_b1ka_train+ceq_b1ka_train:xf_test_ka+f2_test_ka+sib_test_ka+annf_test_ka+ctx_b1ka_test+cev_a1_test+sup3_ka_test+cem_b1ka_test+ceb_b1ka_test+cer2_b1ka_test+ceq_b1ka_test tag=fin5sa
   $PY src/blend.py tags=fin5sa+fin5b out=fin5s margin=0.3 grid=1
   ```
3. **Teammates' members.** Their raw stage-2 scores arrive as `oof_<tag>.parquet` / `pte_<tag>.parquet` (columns id1,
   id2, p), gated with `ana/kcheck.py` and stored as `kcxQ` and `kfx` in `ER_WORK`. Their training code lives in
   their own repositories, not in this one.
4. **Team blend and outputs:**
   ```bash
   $PY ana/wblend.py tags=kcxQ+fin6s+kfx out=tb3 step=0.25 fine=0.05 unseen=equal   # weight search -> 0.7 / 0.3 / 0.0
   $PY ana/tb5.py                                                                     # same weights, France member fin5s
   $PY ana/qfix.py apply tag=tb5 out=tb5_frm2 fr=-2 cand=pte_tb5
   $PY ../student_resource/utils/validate_submission.py --matching $ER_OUT/tb5_frm2/matching_results.tsv \
       --candidate $ER_OUT/tb5_frm2/candidate_pairs.tsv --test-dir $ER_DATA/test
   ```
   The weight search can use `fin6s` in place of fin5s: the two are identical on tsim and on US/India test rows, and
   differ only on France, which gets fixed equal weights.
5. **Stored outputs:**
   - `versions/tb5_frm2/` holds the outputs, `meta.json` and a code snapshot.
   - `matching_results.tsv` has 5,828,829 links, and the official validator passes.

## 11. Compliance

- **No external data, APIs or lookups.** Normalisation rules are hand-written (abbreviations, legal forms, US/Indian
  state tables, French department → region). The Indic transliteration dictionary and the city → region map in `fx`
  are learned from the provided records.
- **Pretrained models.** All are open-licence (MIT / Apache-2.0) and under the 8B cap; the largest is Qwen3-Reranker-4B
  at 4.0B. LightGBM is MIT.
- **Unlabelled test records** are used as inference inputs: IDF statistics, competition features, sibling expansion
  and ANN. For France, they are also used for self-training: pseudo-labels in `cef`, `ceqf`, `cfr`, `cxf` and cxQ's
  extra rows. The organisers confirmed that self-training on the provided test records is allowed. No test labels
  are used anywhere.
- **Country** is never a model feature. It is used for blocking within a country, for the unseen-country blend
  weights, and for the calibration shift. The set of unseen countries is derived from the training file.
