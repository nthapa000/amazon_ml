# plan1.md — closing the gap to the top 3 (Amazon ML Challenge 2026, Business Entity Resolution)

> **Part IV (very bottom, 18:15 IST) = what the organizers' own Q&A answers add.** It gives two new levers (IV.2 L1
> cross-script matching for India, L3 per-country stage 2), a twin check (L2), and a compliance/packaging checklist
> (L5) that must be done before the zip.
>
> **plan1.md = plan.md (Parts I and II, unchanged below) + Part III** (bottom), which reviews
> `strategies_for_winnig.txt` item by item against what the pipeline already does, and keeps only what is new.
> **Short verdict:** the file is a generic entity-resolution playbook. About 90% of it is already built, and much of
> the rest has been measured and rejected. It adds **five small, cheap items** (III.2). None of them changes the
> Part II priorities (N1 merge of the two machines' stacks, cfs + aux, N2/N3).

> **Read Part II first** (bottom of this file, written 17:30 IST after `analysis_summary`). It supersedes Part I
> wherever they disagree:
> - A2 (CE-competition features) is demoted: a linear probe measured +0.00003.
> - A4 (re-probing the France shift) is dropped: the δ sweep is flat from −1 to −3.
> - The France share of CE training data is ~67%, not 15%.
> - §4 S2↔S3 mining, the empty-address recall track and MLM adaptation are measured dead ends.

Written 2026-09-27 17:05 IST from `approach.md`, `EXPERIMENTS.md`, `VERSIONS.md`, `README.md`, `CHANGES_kushal_local.md`
and the code in `er/src/`. The dataset, the Q&A PDF and `analysis_summary.pdf` were **not** available to this session, so
every data claim below comes from the experiment log. Section 6 lists the EDA to run on the cluster to confirm them.

---

## 0. Reality check (read first)

| | value |
|---|---|
| Our best local (tsim, US+India only) | 0.9912 |
| Our best public LB | 0.987862 |
| Rank 3 / 2 / 1 | 0.990810 / 0.991043 / 0.991483 |
| Gap to rank 3 | **+0.00295 LB** |
| Time left (deadline 23:59 IST per `approach.md`) | **~7 h**, and at most 5 uploads today (fewer if some are used) |

**The LB decomposes as `LB ≈ 0.8503·F(US+India) + 0.1497·F(France)`** (France = 15% of test S1, measured with the
`probe_c6_noFR` upload).

- Our France: (0.987862 − 0.8503·0.9912) / 0.1497 ≈ **0.969**.
- Rank 3 needs France ≈ **0.9886** at our US+India, i.e. **+0.020 France F**.
  - Every +0.001 on US+India is worth only +0.00085 LB.
  - The US+India headroom that can realistically be closed today is ≤ +0.0005.
- **The gap to the leaders is France, not US/India.** Leaders at 0.991 imply their France is ~0.99, as good as their
  US/India. So France is not intrinsically harder: **our France is miscalibrated / under-adapted.** This is the single
  most important conclusion, and it drives the priorities below.

**About the request for "a really big LLM":** not tonight. `ceq` (Qwen3-Reranker-4B LoRA) took ~30 GPU-hours to train
and score. A 7–8B model would take 2× that, and the 8B cap excludes Qwen3-Reranker-8B (8.19B). The zero-shot LLM judge
already failed (AUC 0.53, EXPERIMENTS.md). The big-model budget is better spent **continuing the models we already have
on France-shaped data** (§3, B1), which takes ~1 h of training per half. If the deadline is actually later than tonight,
see §8.

Realistic expectation for tonight: **+0.0005 to +0.0020 LB** (→ 0.9884–0.9899) if the France levers land. Rank 3 needs
everything below to work *and* France to respond like it did to `ceb` (+0.009). Treat top-3 as the stretch goal and
top-~30 as the likely outcome.

---

## 1. Where every point is lost

| component | now | ceiling / target | lever class |
|---|---|---|---|
| US+India matcher quality | 0.9912 | 0.99666 (perfect matcher on existing candidates) | CE-competition features, blends, per-country decision |
| US+India blocking recall | recall 98.84% at p1 ≥ 0.005 | +≤0.0005 | **skip tonight** (10–14 h rerun) |
| France calibration | δ = −2 fitted to the old `fin` model | re-fit for the current model | label-free moment matching + 1–2 LB probes |
| France text understanding | cef / ceqf (weaker base or uncertain) | cer2 continued on France-shaped data | GPU, ~3 h end to end |
| Local ↔ LB gap | local has no France | a France proxy in the local score | synthetic France set (§4) |

---

## 2. What the organizers' Q&A allows (as recorded in EXPERIMENTS.md, 26 Sep 23:10)

- Self-training / pseudo-labels on **test inputs**: allowed.
- Synthetic pairs generated from the **provided records** (train + test): allowed.
- A stricter threshold for countries unseen in training: allowed. Keep it data-driven via `unseen()`; never name a country
  in code.
- **Private ranking = private score of your best *public* submission.** Every upload must be package-consistent: its
  candidate file must contain all of its matches.
- Candidate size = total pairs; a smaller set ranks higher on ties. Keep `cands.py thr=0.005`.
- Not allowed: external data, geocoding, business lookups, or LLM-generated business knowledge. Synthetic data must be
  built by rule/statistics from the provided records only (§4).

---

## 3. Tonight's plan (priority order, timed)

Start the long GPU job **first**, then do the CPU levers while it runs.

### B1 — `cer2f`: continue the strongest CE (bge-reranker-v2-m3) on France-shaped data  ★ biggest expected France gain
Why: every France jump so far came from raw-text multilingual CE quality (ceb gave France +0.009, cx +0.0025, ANN
+0.007). `cef` continued the **weaker** `ceb`, and `ceqf` continued the 4B model on the older set. The strongest encoder,
`cer2` (band AUC 0.9839), has never seen French-shaped pairs.

Steps:
1. **Build `fra3`** with `fradapt.py`, with three changes:
   - **Teacher** = the current best stage-2 (fin5/fin6, whatever produced 0.987862), not `fin2`.
   - **Tighten the hard negatives** (bug noted in approach.md §7: `max(token_set) ≥ 80` lets ~72% of the pool through):
     - keep a negative only if name token_set ≥ 80 **and** address token_set ≥ 80 **and** they are not identical; or
     - it matches the "same address, exactly one core word differs, and that word is *not* a learned generic word"
       pattern (the confirmed French distractor, match rate 0.07–0.14 on tsim).
   - **Learned noise, not hand-written.** Estimate operation frequencies from train GT pairs (§4) and sample `syn`
     mutations at those rates, using French vocabulary learned from the French test records (`generic_suffixes()`
     already does this for last words).
2. **Mixture:** match the test country mix as asked.
   - Replay US/India band pairs at India 47% : US 38%, and French rows (pos/neg/s1neg/syn) at 15% of the *effective*
     sample — but at least 300k French rows, because that is what adapts the model.
   - `cef` used ~50% French with no US/India regression (band AUC 0.9806 vs 0.9804), so if in doubt err toward more
     French, not less.
3. **Train:** `ce.py init=cer2_b1ka extra=fra3 txt=raw bf16=1 epochs=1 lr=1e-5 half=0|1`, one Blackwell GPU per half,
   in parallel. Use a lower LR than the original to avoid forgetting.
   - **Gate:** US/India band AUC must stay ≥ 0.9835. If it drops, do not use it.
4. **Score:** `unseen=1` re-scores only the French band pairs (~0.9M, ~1 h per half, parallel), then `merge unseen=1
   base=cer2`.
5. **Patch** into the existing stage-2 fold models, fin6-style: `train.py … models=<best> alias=cer2f:cer2 tag=fin7`.
   No retraining; US/India rows are byte-identical.
6. **Re-tune the France δ** for fin7 (A4 below) and upload.
7. **Timing:** set build 15 min, train ~1 h, score ~1 h, patch 30 min → **ready ~20:30–21:00 IST**. Start by 17:45 or
   drop it.

### A1 — per-country decision parameters (CPU, ~20 min)  small, safe
- `decide.tune` fits one global (margin, m, a) on tsim. US and India behave differently (US 0.99070 vs India 0.99173), so
  fit the grid **per country** and store `dec_<tag>.json` as `{country: cfg}` with a fallback to the global cfg.
- `unseen()` countries (France) keep the global cfg plus δ.
- Also widen the grid a little: margin {0.15, 0.2, 0.25, 0.3}, a {0.9, 1.0, 1.1, 1.25}, floor {0.01, 0.02, 0.05}.
- Expected **+0.00002–0.0001 OOF**. It touches only US/India rows, so its LB delta is directly readable.

### A2 — CE-competition features in stage 2 (CPU, ~1.5 h incl. retrain)  most promising US/India lever
`c_marg2` (stage-1 p margin over the runner-up S1 for the same S2/S3 record) carries 47–77% of stage-2 gain, but the
same margin has **never been computed on the cross-encoder scores**, which are much sharper than stage-1 p.
- New `ctxce.py`: for each CE column in {cer2, ceq}, over pairs that have a CE score:
  - `x_marg2 = ce − max(ce of other S1 for the same id2)`
  - `x_r2` = rank of the pair within its id2
  - `x_r1` = rank within its id1
  - `x_gap1 = ce − max ce within id1`
  - `x_n2b` = number of S1 candidates for this id2 with ce > 0
  NaN outside the band; LightGBM handles it.
- Contrast features against the runner-up S1 (self-join on `c_oth2`): the difference in `co_ts`, `ad_ts`, `hn_conf`
  between this pair and the id2's runner-up pair.
- Retrain the best stage 2 with `extra=…+ctxce_*` (Optuna params, ~40 min).
- **Adopt if OOF ≥ +0.00005 and ≥ 3 of 5 folds are positive** (use `ana/cmp_folds.py`). The usual 0.0001 bar is too
  strict for tonight; the LB is that tight.
- Expected **+0.0001–0.0003 OOF**. It may also help France, because margins are self-normalising across a country's
  shifted CE scale.

### A3 — blend every stage-2 variant we have (CPU, ~30 min)
- Members: fin3, fin4, fin5 (+ceq), fin6 (ceqf patch), cx1 (kushal's bge-v2-m3, separately trained, so a different
  error pattern even though it is the same base), and fin7/A2 when ready.
- `blend.py grid=1` only searches "first vs rest". Replace it with a coordinate-ascent or scipy weight search on OOF
  logits, with 5-fold CV of the weights to avoid overfitting.
- b34 gave +0.000007. Adding models trained with *different CE columns* (ceq vs cer2 vs cx) is where the diversity is.
  Expected +0.00002–0.0001.

### A4 — France δ (and slope) by label-free moment matching, then confirm on the LB
Evidence: on `fin`, the LB optimum δ = −2 was exactly where France's predicted-empty rate equalled US/India's (5.78%).
If the data generator is country-agnostic (check in §6, E3), then France's true matches-per-S1 distribution equals
US/India's. So:
- For any new model, choose δ (and optionally a slope k in `p' = σ(k·logit p + δ)`) so that France's test **histogram
  of predicted links per S1** (0, 1, 2, 3, 4+) and **mean Σp per S2/S3 record** (owner rate) match US/India test's. This
  needs no labels and no upload.
- Upload the matched δ and at most one neighbour (±0.5). Break-even analysis from the fr=-1/-2 probes says the curve is
  flat near the optimum, so do not spend more than 2 slots here.
- Also try a France-only one-to-one margin of 0.3. French distractors are near-duplicate S1s at the same address
  (21k same-address S1–S1 pairs), where a larger margin protects precision. Use the same moment-matching test and only
  upload if it changes more than 1% of French rows.

### A5 — micro-fixes worth a few 1e-5 each (bundle into whichever upload is next)
- **Second-chance reassignment:** if an id2's argmax S1 drops it in `pick_ef` but the runner-up S1 had p ≥ 0.9 with a
  margin, give it to the runner-up. Test on OOF; adopt only if positive.
- **Candidate threshold:** keep `thr=0.005`. 0.01 costs −0.0001 OOF, and the leaderboard is score-first.
- **Seeds:** stop. Measured noise is 0.000001–0.000004 and they add nothing.

### Upload schedule (assuming 5 left; adapt if fewer)

| # | ~time IST | content | isolates |
|---|---|---|---|
| 1 | 19:00 | current best + A1 (per-country decision) + A3 blend | US/India gain (France rows unchanged) |
| 2 | 20:00 | + A2 CE-competition stage 2 (if the OOF gate passes) + A4 moment-matched δ | model + France calibration |
| 3 | 21:15 | + B1 `cer2f` France patch (fin7) + re-matched δ | France text adaptation |
| 4 | 22:00 | best of the above with δ ± 0.5 or a France margin of 0.3 | France calibration fine-tune |
| 5 | reserve | keep for a fix or a rollback | — |

**Freeze at 22:30.** Build the zip from the configuration with the **best public score**, because the private rank uses
that one. Run `validate_submission.py --check-ids` and confirm the candidate set is `cand_test_final` (the pruned set).

---

## 4. Synthetic data from the train + test corpus (the "learned generator")

Goal: France-shaped **labelled** pairs, built only from the provided records, for (a) CE adaptation (B1) and (b) a
local France proxy score (§5).

### 4.1 Learn the noise process from train GT (US+India, labelled)
For every GT pair (S1, S2/S3), diff the two records and count operations. Name and address are handled separately, and
the rates are conditioned on source (S2 vs S3), because the two sources are generated differently.

| field | operations to count |
|---|---|
| name | legal form changed / dropped / expanded; last word dropped (truncation); generic word added / swapped (learned list); token reorder; adjacent-char transposition / deletion / substitution; case change; accent strip; `&` ↔ "and"; DBA replacement (low similarity); domain-style name |
| address | component dropped (postcode, state, city); reordered; street-type abbreviation; house-number format (N°, leading zeros, suffix bis/b); landmark added ("Near …"); empty address |

Output: `noise_ops.json` with P(op | source, field) and P(k ops per record).

### 4.2 Learn the distractor process from train (unlinked S2/S3 = ~2.7M)
For each unlinked record, find its most similar S1 (existing candidates + stage-1 p). Classify the relation:
- same address + one core word swapped (non-generic)
- same core name + different street / house number ±1
- invented DBA name at an S1's address
- fully unrelated

This gives P(distractor type) and **distractors per S1** per country. Check that it is country-invariant between US and
India (E3).

### 4.3 Apply to the French test S1 records (no French labels needed; exact by construction)
- **Positives:** a French S1 → a clone mutated with the 4.2 operation rates, using French vocabulary learned from the
  French test S2/S3 records:
  - legal forms: SARL / SAS / SASU / SCI / EURL / SA / EI / Ets
  - street types: R. / Av. / Bd / All. / Imp. / Rte / Ch.
  - generic suffixes: groupe / développement / holding / participations / associés / international
  - department ↔ region swaps using the existing rule table
- **Hard negatives:** (i) another French S1 at the same normalized address, noised the same way (exact label, because
  S1 is deduplicated); (ii) a French S1 with one *non-generic* core word replaced by a word from the same slot of
  another French S1 at a different address; (iii) the same name with a different house number or street.
- **Proportions:** France ≈ 15% of the combined training rows, as asked, matching the test mix
  (India 47 / US 38 / France 15). Also produce a France-heavy variant (~40%) and pick by the §5 proxy.
- **Leakage guard:** synthetic pairs are only for CE training / proxy scoring; never add them to the tsim OOF used for
  decision tuning.
- **Validity checks before use:**
  1. Train a small classifier to separate real French S2/S3 from synthetic French S2/S3 (adversarial validation).
     AUC ≤ 0.65 means realistic enough; higher means inspect the top features and fix the generator.
  2. Apply the same generator to *US/India* S1 and check that stage-1 p on those synthetic positives has a distribution
     similar to real US/India positives (KS test).

---

## 5. Closing the local ↔ LB gap

Local 0.9912 vs LB 0.9879 is almost entirely **France missing from the local metric** (US+India LB-implied ≈ local).
- Define **local LB proxy = 0.8503·F(tsim) + 0.1497·F(France-proxy)**. Log it in EXPERIMENTS.md for every candidate.
- **France-proxy (tonight, cheap):** score the §4.3 synthetic French *pairs* with each model's CE / stage-1, and report
  pair-level AUC and precision at the chosen threshold.
  - This is not entity-level macro F, but it ranks France changes. The label-free self-check (`selfcheck.py`) could not
    do that: it rated the fr=-2 shift as harmful although the LB showed it helps.
- **France-proxy (full, only if the deadline allows, §8):** build a complete synthetic French *world* (French S1 +
  synthetic S2/S3 with distractors at the learned rates), run blocking → stage-1 → CE → stage-2 on it, and get a real
  macro F0.5 for France offline.
- **Calibration check:** after each release, compare France vs US/India test on predicted-empty rate, links/S1,
  owner rate, and the share of uncertain pairs (0.1 < p < 0.9; France 0.41/S1 vs US 0.14 on `fin`).
  - A drop in France uncertainty with the moments matched is the best label-free sign of progress.

---

## 6. EDA to run on the cluster (train + test) — 30 min, CPU, polars

Run these before or alongside §3. Each one either confirms an assumption above or changes a decision.

| # | question | why it matters |
|---|---|---|
| E1 | Per country and split: #S1, #S2, #S3, (S2+S3)/S1, empty-address rate, name token count, script mix, legal-form frequencies | test France vs US/India shape; checks the generator is shared |
| E2 | Train GT per country: histogram of matches per S1 **per source** (S2 count, S3 count), singleton rate, unlinked share of S2/S3 | if the max matches per source is small, a per-source quota rule is a free precision gain |
| E3 | Are E1/E2 distributions the same for US and India? (chi-square on matches-per-S1 histograms) | if yes, France's true distribution is the same, which justifies A4 moment matching |
| E4 | Top 50 last words and first words, S2/S3 vs S1, per country (test included) | the generic-word lists for §4 and the French vocabulary |
| E5 | 4.1 operation rates by source | generator parameters |
| E6 | 4.2 distractor-type mix by country; for test, estimate with model p | whether France has more same-address distractors, which argues for a France margin / δ |
| E7 | Test: France vs US/India on predicted links/S1, Σp per record, uncertain-pair share, for the current best model | baseline for A4 |
| E8 | Residual errors of the best model on tsim by type: blocked, below threshold, wrong merge, singleton FP, each × country × name commonness × empty address | confirms where the US/India 0.0055 headroom is |
| E9 | Test-side transliteration coverage: India native-script tokens not in the learned dict (test 96.4% vs train 97.8%) | India lever for §8 (dict extension via confident test matches) |

Sketch for E2/E3:
```python
import polars as pl
gt = pl.read_csv('dataset/train/train_ground_truth.tsv', separator='\t').with_columns(
    ids=pl.col('matched_entity_ids').fill_null('').str.split(',').list.eval(pl.element().filter(pl.element() != '')))
s1 = pl.read_csv('dataset/train/train_source1.tsv', separator='\t', columns=['entity_id', 'country'])
g = gt.join(s1, left_on='source1_entity_id', right_on='entity_id').with_columns(
    n2=pl.col('ids').list.eval(pl.element().str.starts_with('S2-')).list.sum(),
    n3=pl.col('ids').list.eval(pl.element().str.starts_with('S3-')).list.sum())
print(g.group_by('country', 'n2', 'n3').len().sort('country', 'len', descending=[False, True]))
```

---

## 7. Measured dead ends — do not spend time on these tonight

From EXPERIMENTS.md and CHANGES_kushal_local.md:
- decision-rule grid (global, saturated)
- `xf2` / `xf3` / `feats3` hand features (+0.00000–0.00007)
- entity gate (+0.00001)
- XGBoost member / blend
- stage-2 seeds (noise ~1e-6)
- zero-shot LLM judge (AUC 0.53)
- a second CE of the same family *as a feature* (corr 0.984)
- expansion round 2 (0.6% of misses)
- K200 caps
- `ctx` recomputed from stage-2 scores (u5, no gain)
- duplicate-twin listing rule (+0)
- label-free `selfcheck` F_est for ranking France changes (contradicted by the LB)
- any blocking/ANN rebuild or normalization rerun (9–14 h)

---

## 8. If the deadline is later than tonight (> 24 h left)

In order:
1. **Full synthetic France world** (§5): real offline France macro F. Then tune δ / slope / margin / ef for France
   without LB probes.
2. **Stage-2 trained with the France world included** (labelled synthetic France pairs with their full candidate
   context), so the GBDT learns France's CE-score scale instead of relying on a post-hoc δ.
3. **Normalization fixes** (FR frères/frs, apt, bis/b house numbers, US St/Saint and CT/MT/NE/WY), then a full rerun.
   Audited at about +0.00007 LB; only worth it inside a rerun done for other reasons.
4. **India transliteration dict extended** from confident test matches (native token ↔ Latin token alignment,
   self-training allowed). Closes the 96.4% vs 97.8% coverage gap.
5. **Bigger CE, only if the above are done:** ceq retrained from scratch on (train band + France world) mix. The
   ≤ 8B, Apache/MIT options are Qwen3-Reranker-4B (already set up) or a Qwen2.5-7B LoRA scorer (~60 GPU-h; only with
   ≥ 3 days).
6. **Recall:** ANN top-5 (recovers 46.5% vs 43.5% of blocked misses) only if the France-world run needs the ANN
   rebuild anyway.

---

## 9. Decision rules for adoption tonight

- US/India changes: adopt if the OOF gain is ≥ +0.00005 **and** at least 3 of 5 folds are positive. Every measured
  OOF gain has shown up on the LB at 1–3× (larger for multilingual-text changes).
- France changes: adopt only after an LB upload that differs from the current best **only in French rows**, so the
  delta is attributable.
- Never upload anything without `validate_submission.py --check-ids` PASS, with matches ⊆ candidates.
- The final zip must be the configuration of the best *public* score, reproducible by `run_all.sh`, with the
  Documentation updated (synthetic data, self-training, δ method, model licences / params).

## 10. Open items for the team

1. Which tag produced 0.987862 (fin5? fin6? b34c?) and how many uploads are left today. This decides the teacher for
   B1 and the upload schedule.
2. Are both Blackwell GPUs free right now? B1 must start by ~17:45 IST.
3. Please re-attach the Q&A PDF and `analysis_summary.pdf` if any organizer answer changes §2 (for example a limit on
   test-derived synthetic data).

---
---

# Part II — after `analysis_summary` (written 17:30 IST, ~6.5 h to deadline)

## II.0 What the new measurements change (facts only)

| fact (from analysis_summary) | consequence for the plan |
|---|---|
| Train US and India have **identical generator constants**: 5.58% singletons, 3.461 matches/S1, the same per-source histograms (S2 at most 5, S3 at most 6, 80.5% of S1 matched in both sources) | France almost certainly shares them, which gives exact **label-free anchors** (II.2 N2, N5) |
| US+India loss: 37.6% never blocked, 62.4% scored too low; **86% of the loss is under-prediction** | the precision side is saturated (0.9987); recall on contested pairs is where the gain is |
| Of the 87,685 in-set true pairs we miss: **47% lost at one-to-one** (another S1 scored higher), 33% lost to margin 0.3, 20% lost in pick_ef; median p 0.32 | the biggest single pool is **S1-vs-S1 contests for the same record**, not thresholds (II.2 N3) |
| Oracle-k 0.99309 vs 0.99112; every *global* rule variant is within ±0.00003 | only rules that add *new information* can move it (count prior, duel model), not re-tuning |
| France: **8.18 candidates/S1, 3.52 band pairs/S1 (2.4× US/India)**; 10.7% of French S1 share an address with another S1 (4–5% US/India) | France is a *discrimination* problem at shared addresses |
| Stage 2 is country-agnostic and tsim has no France, so **LightGBM weights a France specialist from US/India rows only** | a France CE only pays if it keeps US/India band AUC ≥ 0.9835 **and** ships with `aux=` French rows |
| cxf beats cx by +0.00025 band AUC overall but **+0.00098 on the pairs we get wrong** | French adaptation targets exactly the contested cases, so keep investing there |
| Data: S2 upper-cases 14–23% of names; S2/S3 inject **spurious accents** (France 24%, US 6.7%); 22.7% of India S2 names are fully native script | cheap **text-view TTA** for CE scoring (II.2 N4) |
| Two machines hold two nearly independent CE stacks: Kushal has cx / cxf / qw, fx; m25csa019 has cer2 / ceq (4B) / cef / ceqf, fin5 / fin6 | the **cross-machine merge** is the cheapest real diversity left (II.2 N1) |

**Measured dead — do not spend time here tonight:**
- empty-address recall recovery: break-even precision 0.45, best achievable 0.32; every variant negative
- S2↔S3 cross-source mining: 1.07× lift in regime; *anti-informative* in the band
- MLM domain adaptation: bge-v2-m3 ships with no LM head, and there is no MLM code
- France global δ re-probe: plateau between −1 and −3
- decision-rule re-tuning, isotonic calibration, soft one-to-one: all within ±0.00003
- `ew=1`: −0.00003
- `xf3`, `pc_eq`, `st_eq` fixes
- a 4B model *trained from scratch tonight*: scoring alone takes ~4.4 h per half

## II.1 Already in flight (from the team's plan; keep it)
qw merge → **fxq**, then cfs France specialist (warm start from cxf, ~67% French rows, `txt=raw`, matched halves) +
`aux=s2mat_test:frlab_test` → **fxc**, then seeds and blend. Everything below is **additive**. Nothing here needs
the GPUs before cfs finishes (~18:40).

## II.2 New ideas, ranked by expected LB gain per hour (be ready to kill any of them)

### N1 — Merge the two machines' stacks  ★ best EV, CPU only, ~1 h
The team has two stage-2 families that barely share CE inputs:
- **fx family** (cx, cxf, qw)
- **fin5/fin6 family** (cer2, ceq = Qwen3-Reranker-**4B** decoder, cef, ceqf)

Every time this project added *architectural* diversity it paid (ceb, cx, ANN). The 4B decoder is the most different
model the team owns, and it is sitting on the other machine.

(a) **Logit blend of stage-2 outputs:** `blend.py tags=fx(or fxc)+fin6 split=tsim grid=1`.
  - Both were built on the same imported `s2mat` / tsim split, so `(id1,id2)` keys should align.
  - **Gate:** assert ≥ 99.9% key overlap on tsim and test before trusting the OOF.
(b) **One stage-2 with every CE column:** `extra=…+cer2_b1ka_*+ceq_b1ka_*+cef_b1ka_*` (test side: `ceqf` patched in
  for unseen-country rows, as fin6 did).
  - Check first that ceq's band is the same b1ka band (lo 0.003, hi 0.997). Otherwise NaN-fill; LightGBM handles it.

Only needs copying ~10 parquets (oof/pte/CE columns, a few hundred MB). **Adopt** the better of (a) and (b) if OOF is
≥ +0.00005. Expected **+0.0001–0.0004 OOF**, and more on the LB if ceq/ceqf carry French signal (the France
amplification has been 3–10×).

### N2 — Generator count prior in the decision  (CPU, ~40 min, measurable)
`pick_ef` treats pair probabilities as independent and knows nothing about how many matches an S1 usually has. The
generator does: the *k-th link hazard* P(K ≥ k+1 | K ≥ k) is 0.944, 0.943, 0.809, 0.666, 0.543, 0.439, 0.347, 0.268…
(from the train histogram, identical for US and India). Per source there are hard caps: S2 ≤ 5, S3 ≤ 6.
- Replace `e0 = Π(1−q)` and the set score with a prior-weighted expected F: P(K = k) ∝ prior(k) × likelihood from the
  sorted q. Also enforce the per-source caps.
- It is also a label-free check on France: France under-predicts (3.36 links/S1 vs 3.46).
- This is *new information*, not a re-tune, so the saturation result does not rule it out.
- **Kill** if tsim gain is < +0.00003.

### N3 — Duel model for contested records  (CPU, ~1.5 h, measurable)
41k of our misses are records that went to the wrong S1.
- For every id2 with ≥ 2 candidate S1s, build pairs (A = best, B = runner-up) with **difference vectors** of *all*
  pair features, including every CE column and address/house-number evidence. Train LightGBM on
  "does A really own x?" (5-fold, same folds).
- Feed `duel_p` back as a stage-3 feature, or re-rank one-to-one with it.
- Variant: LightGBM `lambdarank` with `group = id2` (listwise over competing S1s, plus a "none" option).
- This differs from the rejected "CE-margin" probe: that used one score's margin, this learns interactions across
  ~120 feature differences.
- **Kill** if < +0.00005.

### N4 — Text-view TTA for the cross-encoders  (GPU after 18:40, ~45 min per view for one model)
Score each band pair under extra views and average the logits:
1. **Symmetric:** (t2, t1) as well as (t1, t2). Cross-encoders are order-sensitive.
2. **Accent- and case-folded:** unidecode + lowercase both sides. S2/S3 inject spurious accents (France 24%) and
   upper-case (S2 ~20%); S1 never does.

To keep stage 2 consistent, TTA must be applied to **both** the tsim OOF band and the test band. Use the strongest
single column (cxf or cfs), replace it with its TTA average, then retrain stage 2 (~25 min).
- Expected: small on US/India, larger on France (its accent-injection rate is 4× the US rate).
- **Kill** if US/India band AUC drops.

### N5 — Per-source calibration for the unseen country  (CPU, 20 min + 1 upload)
The global δ is at a plateau, but S2 and S3 are *different noise processes*: S3 rewrites addresses (address Jaccard
0.48 vs 0.64 for S2) and truncates names more.
- **Label-free diagnostic:** compare France's predicted per-source histograms (share of S1 with 0 S2 links, 0 S3
  links, mean S2/S1 and S3/S1 links) with the invariant constants 13.04% / 12.07% / 1.674 / 1.788.
- If one source deviates much more (likely S3), fit a **(country × source) δ** so each France histogram matches its
  constant, and upload once.
- Generic via `unseen()`, never naming France; organizer-allowed ("stricter threshold for unseen countries").

### N6 — Per-country quantile alignment of CE columns  (CPU, ~40 min + 1 upload)
French CE scores sit on a shifted scale (ceb French median −8.1 vs −3.1 on train). Stage 2 learned thresholds on
US/India scale.
- Add rank-within-country versions of each CE column (unsupervised test statistics, which are allowed) and retrain
  stage 2.
- tsim should be ≈ unchanged (US/India ranks are equivalent). The effect lands on France, so test it on the LB only
  if an upload slot is free.
- Risk: if France's true match rate in the band really is lower, ranks over-credit it. The per-source histograms from
  N5 are the sanity check.

### N7 — Cleaner French pseudo-labels by model agreement  (CPU, 15 min)
- Rebuild `frpl` / `frlab` keeping a pair only where **fx, qw and ceq (if merged) agree** (all ≥ 0.97, or all
  ≤ 0.02).
- Drop any S1 whose pseudo-positive count breaks the generator caps (> 6 per source, or > 11 in total).
- Use it for the aux rows of the final stage 2, and as extra rows for any later cfs continuation.
- Different model families disagreeing is the cheapest label-noise detector available.

### N8 — Unseen-country indicator + aux rows  (CPU, ~40 min, 1 upload, *risky*)
Organizers allow country as a feature when treated as an open set. Add `unseen = country ∉ train countries` and
train with `aux=` French rows, so stage 2 can learn **French-specific weights for the CE columns** (cfs especially)
instead of US/India weights.
- Circularity caveat: aux labels are our own predictions. It only helps because cfs/cxf carry *new* French signal
  from exact-label synthetic data.
- Ship it only as its own isolated upload, never merged silently into the best model.

### N9 — Seeds and bagging of the final stage 2  (CPU, parallel with everything)
Two to three seeds of the final configuration, then logit average. Measured noise is tiny, but free while the GPU
runs.

### Stretch (GPU, only if cfs finishes early and N4 is done)
- qw continued on the larger French pool with N7 labels. Decoder diversity, 0.6B, ~2.5 h per half. **Abort** if it
  cannot be scored by 21:00.

## II.3 Revised upload plan (3–4 left; private = best public, so a weaker upload costs nothing but the slot)

| # | when | content | why |
|---|---|---|---|
| 1 | ~18:00 | **fxq** (+ N1a blend with fin6, if the keys align) | biggest cheap, certain gain |
| 2 | ~19:30 | **fxc** = cfs + aux + N7 labels + the winners of N2 / N3 / N1b | France specialist and US/India contest gains together |
| 3 | ~20:45 | #2 + N4 TTA column **or** N5 per-source δ, whichever is ready, isolated so its delta is readable | France calibration |
| 4 | reserve | N6 or N8 as a pure France probe, or a rollback | only if #3 went well |

Freeze modelling at 21:00. Package the configuration with the **best public** score (candidate file = the pruned
cascade set, validator PASS, `frlab` generator script committed, since it is currently missing from the repo).

## II.4 Honest projection
- **Likely:** N1 +0.0002–0.0005 LB, cfs+aux +0.0003–0.0008, N2/N3 +0.0000–0.0003 → **LB ≈ 0.9885–0.9895**.
- **Upside:** ceq/ceqf carry France the way ceb did (+0.009 France F) and N5 fixes a source-specific France bias →
  **~0.990–0.991**, which is top-3 territory.
- Rank 3 needs France to rise ~+0.015–0.02 F. The ideas most likely to move France that much are N1 (the 4B decoder
  stack), cfs, and N5, in that order.

## II.5 If the deadline turns out to be later
In priority order:
1. France synthetic *world* (full pipeline on synthetic French S2/S3 with the learned operators and distractor mix)
   → real offline France macro F.
2. Stage 2 trained with that world included.
3. ceq continued on France data (cef-style), full two halves.
4. Normalization gaps (US St/Saint, CT/MT; FR frères/apt/bis) inside that rerun.

---
---

# Part III — review of `strategies_for_winnig.txt` (written ~18:00 IST)

## III.0 Verdict in one paragraph
The document is a sound, *generic* entity-resolution recipe:
- normalize → multi-channel blocking → pairwise GBDT → margin features → F0.5 decision → singletons → S2/S3
  consistency → ensemble;
- it ranks cross-encoders and dense retrieval as low priority ("Priority 4").

Our pipeline already does every Priority 1–3 item and, beyond that, the Priority 4 items, which are where every
large gain of the last two days came from:

| lever | OOF gain |
|---|---|
| ANN | +0.00287 |
| raw-text multilingual CEs | +0.00072 / +0.00014 / +0.00016 LB ×3 |
| cascade stage 2 | +0.00015 |

The document contains **no information about this dataset**: nothing on France, the 15% unseen country,
source-specific noise or the measured loss decomposition. Its advice to "prioritise blocking and normalization over
cross-encoders" is **the opposite of what this dataset has shown**:
- blocking-side headroom is 0.00334, but it cannot be harvested at F0.5 (break-even precision 0.45 vs best 0.32);
- the reranker ladder moved France +0.009.

So: use it as a checklist confirming nothing basic is missing, not as a new direction.

## III.1 Item-by-item audit (doc section → our status → evidence)

| doc § | recommendation | our status | evidence / note |
|---|---|---|---|
| 1, 18 | optimize entity-level macro F0.5, never 0.5 | **done** | `metric.py`; expected-F0.5 set selection (`decide.py pick_ef`) tuned on OOF |
| 2, 24 | candidate recall first; measure Recall@K | **done** | blocking b1→b3→K100→expansion→ANN: pair recall 0.921→0.9809+, ceiling 0.99666; `ana/cands.py` recall per cut |
| 3 | multiple blocking channels (exact name, char TF-IDF, tokens, address, name×city/hn/PIN) | **done (mostly)** | 11 key families (tok, pair, tokst, cat, full, nw/ww/hw/fw, code, sub) + sibling expansion + e5 ANN. Char TF-IDF / BM25 not built as channels, see III.3 |
| 4 | strong normalization | **done** | `norm.py`: legal forms, street types, states, accent strip, junk prefixes, France rules |
| 5 | learn abbreviations from training pairs | **partial** | transliteration dict learned (`dicts.py`); abbreviation *mining* not built, see III.2-E |
| 6 | Indic transliteration, learned from train | **done** | learned token dict (1347 name tokens); coverage train 97.8% / test 96.4% |
| 7 | structured address features | **done** | hn_inter / hn_conf / hn_sub, pc_eq, st_eq, ad_ts / jac / widf / cov, geo.py city→state |
| 8–9 | GBDT on rich pair features | **done** | 49 + 7 + 6 + 13 + 15 features; LightGBM stage 1 + cascade stage 2 |
| 10 | score margin, rank, close candidates | **done** | `ctx.py`: c_marg2 (the #1 feature, 47–77% of gain), c_gap1, c_r1 / c_r2, c_n1h |
| 11–12 | hard-negative hierarchy | **done for GBDT/CE; partial for France synthetic** | stage 1 trains on blocking candidates (natural hard negatives); CEs train on the uncertain band; `fr_aug.py` builds same-address negatives. Missing levels, see III.2-B |
| 13–14 | entity-level rule with margin + evidence | **done** | one-to-one margin 0.3 + expected-F set rule; every global variant is within ±0.00003 (saturated) |
| 15 | explicit singleton model | **tried, rejected** | `gate.py` AUC 0.99935 but +0.00001; predicted-empty 5.7% vs true 5.58% |
| 16 | S2↔S3 triangle consistency | **done as features; mining rejected** | `sup.py` sibling support (+0.00029 when added). S2↔S3 *label mining* measured anti-informative in the band (0.186 vs base 0.204). Re-computing support from final scores is new, see III.2-A |
| 17 | S1-level validation, slice breakdowns | **done; slices partial** | GroupKFold by S1; `tsim` test-like split (19% of S1 dropped for the orphan rate); adversarial AUC 0.59–0.64. Slice report, see III.2-D |
| 19 | don't hard-code countries | **done** | `unseen()` shift, country never a feature, France rules gated generically |
| 20 | LGBM + XGB + CatBoost ensemble | **XGB tried (par, blend 0); CatBoost untried** | see III.2-C; expected tiny |
| 22 P4 | dense retrieval, cross-encoders | **done, and our main gain source** | ANN e5, six CEs up to Qwen3-Reranker-4B |
| 23 | NAS | **skip** | agree with the doc |

## III.2 What is genuinely new or improvable (cheap, gated, ranked)

**A — Sibling support recomputed from the *final* model**  (CPU, ~40 min + ~25 min retrain; measurable on tsim)
- `sup.py` defines an S1's siblings from **stage-1** scores (`pred=oof_b1ka`, one-to-one margin 0.2, p ≥ 0.9).
  Stage 1 is much weaker than today's stage 2 (band AUC 0.970 vs CE-fed stage 2).
- Re-run `sup.py pred=oof_fx` (tsim) / `pte_fx` (test) with `sib_thr` in {0.8, 0.9, 0.95}. The siblings are then
  more complete and cleaner, which matters most for S1s with several S2 and S3 records.
  - This is the doc's "triangle consistency" in the form that suits this pipeline.
- Leak check: oof_fx is out of fold with the same S1 folds, and siblings are records of the same S1, hence the same
  fold, so this is valid exactly as stage-1 OOF was.
- Also add one "caution" feature from doc §16: `s_iso` = 1 if the pair's S1 has ≥ 2 siblings and x's best sibling
  similarity is below the 10th percentile of true-pair values (x disagrees with every other record of that
  business).
- **Adopt** if tsim gain ≥ +0.00005. Expected +0.00003–0.0002 (the original sup gave +0.00029 on a weaker base).

**B — Complete the hard-negative hierarchy in the French synthetic set**  (CPU, ~20 min; feeds cfs / any French CE)
`fr_aug.py gen` draws negatives by shared address (60%) or first core token (40%). Two levels from doc §11–12 are
missing, and both match the observed French borderline class "(b) same name, different street or house number ±1,
other legal form" (EXPERIMENTS.md, France borderline inspection):
- **L5 same name + neighbouring house number / same street:** take a French S1; keep the name (noised); set the house
  number to ±1–4 or pick another S1's number on the same street; label 0.
  - Exact by construction? **Only if** no other S1 has that name at that address. S1 is deduplicated, so check the
    new (name, address) is not an existing S1 and not the anchor itself.
- **L6 same name + other city in the same region:** swap the city/postcode for another French city of that region
  (from S1's own city list); label 0.

Emit about 150k rows each, in the `(t1, t2, y)` raw-text format. Add them via `aug=` to the next French CE run only.
- **Gate:** US/India band AUC must not drop (cfs acceptance rule, ≥ 0.9835).
- **Caveat:** in train, same-name-other-address pairs are sometimes *true* matches (S3 rewrites addresses). Keep L5/L6
  to ≤ 10% of the French rows so the model does not learn "any address change = non-match".

**C — CatBoost as a stage-2 member**  (CPU, ~45–60 min; low value)
- XGBoost depth 10 gave OOF par and blend +0.00001. CatBoost (ordered boosting, symmetric trees) is a slightly more
  different learner, and the doc's §20 advice is right in principle.
- Try only if CPU is idle: same s2mat features, 5 folds, then logit-blend with fx via `blend.py`.
- **Kill** if blend < +0.00003. Needs `pip install catboost` (Apache-2.0).

**D — Slice-level error report**  (CPU, 15 min; decides where N4/TTA should aim)
The doc's §17 slice list is a good lens we have not fully applied. On `oof_fx`, report macro F0.5, precision and
recall for:
- India S2 names fully in native script (22.7% of India S2);
- upper-cased S2 names;
- domain-style names (3–4%);
- empty S2/S3 address;
- S1 with ≥ 5 true matches;
- S1 whose core name is shared by ≥ 5 other S1s;
- S3 records with address rewrites;
- each by country.

Any slice with a loss share far above its population share is a target for a text-view TTA (N4) or a targeted
feature. This is also the most direct answer to "improve India/US": it shows *which* India/US records still fail.

**E — Abbreviation / variant mining from training pairs**  (CPU, 30 min to mine; *use after the deadline, or only as
a CE text view*)
- Align tokens of matched train pairs; keep frequent (a, b) where b is a prefix, subsequence or known contraction of
  a (engineering↔engg, consultancy↔consult, international↔intl).
- Using it in `norm.py` would need the full 9–12 h rerun, so not tonight.
- Tonight it can only enter as a "canonicalized" text view for the CE TTA (N4), with the same caveat: the view must be
  applied to both the tsim and test bands.

## III.3 Items in the doc that we should *not* do tonight (with the measurement that rules each out)

| doc recommendation | why not |
|---|---|
| more blocking channels (char TF-IDF, BM25, name+city) | 78.8% of never-blocked pairs have an empty S2/S3 address. Measured: adding name-only candidates for them is **net negative** at every precision cut (best −0.00001; break-even precision 0.45 vs achievable 0.32). Also needs the 230 GB `ka` rerun, which does not fit the box |
| separate singleton model | `gate.py` +0.00001 |
| probability calibration / isotonic | re-tested today: within ±0.00003 |
| single global threshold warnings | already entity-level; the grid is saturated |
| XGBoost ensemble | `c2x` par, blend +0.00001 |
| S2↔S3 pair mining for labels | anti-informative in the band (measured) |
| NAS / bigger generic nets | agree with the doc: no |

## III.4 Where this leaves tonight's order (Part II upload plan unchanged, with III.2 slotted in)
1. **N1** merge the two machines' stacks + fxq → upload #1
2. **cfs + aux** (+ **III.2-B** L5/L6 negatives if cfs has not launched yet; otherwise keep them for the next French
   CE) → upload #2, together with the winners of **N2 / N3 / III.2-A**
3. **N4 TTA** aimed at the worst slice from **III.2-D**, or **N5** per-source France δ → upload #3
4. **III.2-C** CatBoost only if CPU is idle; **III.2-E** after the deadline

Expected contribution of Part III on top of Part II: **+0.00005 to +0.0003 LB** (mainly III.2-A and III.2-B).
The projection in II.4 stands.

---
---

# Part IV — hints from the organizers' answers (Query Form Responses, leaders' answers only)

## IV.0 How this was read
- The sheet has 967 queries, of which 342 were answered by the Amazon leaders. The answers reduce to **46 unique
  paragraphs**; 244 answers are "the dataset is on Unstop" and 21 are "AWS is optional".
- Only the **answer column** is used as evidence. Participant questions are quoted only to show what an answer
  refers to.
- **What the leaders did *not* answer is also informative.** XCalibur (25 Sep 14:43) asked directly:
  - is the test set (including France) generated by the same process and noise as train?
  - is the no-match proportion similar?
  - is test S1 deduplicated?
  - do matches always share the country label?
  - how is the public/private split made?

  The reply was the standard rules block and **none of these were answered.** So there is **no leaked hint about the
  generator**: anything we assume about France's generator (for example the country-invariant constants in N2/N5)
  stays *our inference*, not an organizer statement.

## IV.1 Every substantive ruling → what it means for us

| # | leaders' answer (quoted or condensed, date) | status in our pipeline | action |
|---|---|---|---|
| 1 | **"The same business can appear in different scripts across sources. For example, a name may be in an Indic script in one source and in Latin script in another, so your pipeline may need to match across scripts."** (25 Sep 12:07, repeated 19:24, 22:15) | Partly covered. `dicts.py` learns a native→Latin token dict from train pairs; tokens not in it fall back to **unidecode** (crude phonetics). Coverage of native tokens: train 97.8%, **test 96.4%**. The CEs see raw text | **New lever L1** (India) |
| 2 | Transliteration explicitly allowed: **indic-transliteration, anyascii, uroman**, own rule-based code, and **pretrained transliteration models such as IndicXlit** (MIT/Apache, ≤ 8B, offline) (25 Sep 11:44, 12:29 and ~8 more) | we use none of them | feeds L1 |
| 3 | **"Include every Source 2/Source 3 ID you believe refers to the same business … Do not keep only the topmost of content-identical records."** (25 Sep 03:32) | `ana/dups.py`: twin-completion changed tsim by +0.000000 (17 cases). But data shows 2.25% of S2 / 1.62% of S3 rows sit in content-identical groups (**France ~2.9%, the highest**) | **Check L2** (cheap) |
| 4 | Test-set statistics allowed: TF-IDF, token frequencies, blocking index, **"counts of how many Source-1 records share a name"**, **popularity counts**, nearest-neighbour search over test (26 Sep 12:56, 18:43, 19:05) | done: `xf.py` computes commonness per split (test on test), `geo.py` learns city→state from provided records, ANN runs on test | none; cite these answers in the documentation's compliance section |
| 5 | Self-training / pseudo-labels on test and synthetic pairs from provided records allowed (≥ 13 answers) | cef / cxf / cfs, `fr_aug.py`, `frlab` aux | none; confirms Part II |
| 6 | **Country may be a feature, and script detection is allowed**, as long as country is an open set (25 Sep 01:42) | stage 2 is deliberately country-agnostic (`feats.py` line 1) | **New lever L3** (US/India) |
| 7 | **Stricter threshold for a country unseen in training: "Yes"** (26 Sep 19:05, Q4) | fr=−2 via `unseen()` | also covers N5 (per-source δ for the unseen country): same kind of rule, still defined by "unseen", never by name |
| 8 | **Private LB = the private score of your best *public* submission** (25 Sep 18:36; 26 Sep 12:56, 17:03) | known | **L6**: upload every useful variant, since a worse one costs nothing |
| 9 | **"The private leaderboard is scored on the same test files … same three countries … France is in both the public and private splits"** (25 Sep 02:42, 12:07) | known | France δ / France-model effects measured on public **transfer to private**. Public-vs-private variance is sampling only, so do not over-read deltas < 0.00005 |
| 10 | **5 submissions per team per day**; no need to submit every day (25 Sep 07:12) | — | today's remaining slots are the last. Use all of them |
| 11 | Candidate set: **"Total pairs. This is a separate judging criteria beyond the public/private leaderboard score."** (26 Sep 12:56). Early answers said "in a cascade, the input to the first scoring model" (25 Sep 03:14, 06:35); later answers say "the last candidate set fed to your matching model" (25 Sep 17:59, 22:28) and "a pruned later stage is fine" (repeated) | cascade cut p1 ≥ 0.005 → 4.47 / S1 (7.74M pairs) | keep 0.005. In the documentation, **state both numbers** (103 / S1 first pass → 4.47 / S1 scored) and quote the *later* answers, so either reading of the rule is visibly satisfied |
| 12 | Model rules: **per model**, ≤ 8B, MIT/Apache, offline, **fine-tuned only on provided data**; applies to embedder, reranker, matcher **and any preprocessing / augmentation model**; **licence is checked** (≥ 24 answers) | Qwen3-Reranker-4B, bge-v2-m3, bge-base, mMiniLM, MiniLM, e5-small, Qwen3-Reranker-0.6B: all MIT/Apache and ≤ 4B | **L5**: list every model with licence, params and HF revision in the zip (README already has most of it) |
| 13 | **">8B … prohibited even for teacher model"** (26 Sep 18:11) | Qwen3-Reranker-8B (8.19B) was already excluded | L5: state explicitly that no > 8B model was used at any stage |
| 14 | Packages bundling geo/postal/business data are prohibited (libpostal, gazetteers, **"large static country/state/city tables"**); **cleanco banned**; **small hand-written state lists fine, "city-level or postal tables are not"** (25 Sep 01:34, 13:50, 23:14; 26 Sep 00:47) | `norm.py` has hand-written state / region / department / street / legal lists (US_STATES, IN_STATES, FR_REGIONS, _FR_DEP, LEGAL…); **no city table**; `geo.py` learns city→state from provided records only | **L5**: confirm no city-level list crept in (e.g. inside `_FR_DEP` comments or `ana/`), and that `unidecode` (GPL, a library, not data) is acceptable. Keep it, it bundles no geo data |
| 15 | **General-language resources are fine: "fastText, wordfreq fine"** (25 Sep 13:50, 21:37) | not used | **New lever L4** (France) |
| 16 | "You are judged on your results and **reproducible code**"; top packages reviewed | `frlab_test.parquet` has **no generator script in the repo**; `fr_aug.py`, `geo.py`, `package.sh` untracked on Kushal's box | **L5 (must do)** |
| 17 | Methodology document: **no page limit, "prioritize clarity and technical depth"** (25 Sep 03:03) | `Documentation.md` stale | L5 |
| 18 | AI coding assistants allowed; AWS Entity Resolution **prohibited** | — | none |

## IV.2 Actions derived from the leaders' answers (ranked)

### L1 — Cross-script matching for India  (the only *explicit* modelling hint in the whole sheet)
Why it matters: 22.7% of India S2 names and 11.5% of India S3 names contain **no Latin letter at all**, and the
leaders went out of their way to say matching across scripts may be needed. Our stage-1/2 string features see those
names only through the learned dict, and **unidecode** for the ~3.6% of test tokens the dict misses.

1. **Gate first (15 min, CPU):** from the III.2-D slice report, compute tsim F0.5 and missed-pair share for India S1
   whose matches are native-script records.
   - Continue only if that slice's share of the India loss is above its share of India pairs (i.e. the slice is
     under-served).
2. **Feature path (tonight, CPU, ~1 h incl. retrain):** transliterate every native-script S2/S3 *name* with a
   rule-based library the leaders named (**indic-transliteration** or **anyascii**, both offline, no data bundled).
   Then compute `xl_ts`, `xl_jw`, `xl_pr` (token_set, Jaro-Winkler, partial ratio vs the S1 core name) plus
   `xl_skel_eq` (consonant-skeleton equality, which absorbs vowel-length noise such as praaivett ↔ private).
   - NaN for Latin records. Add as `extra=` columns to the best stage 2 and retrain on tsim.
   - **Adopt** if India OOF improves by ≥ +0.00005 and US is unchanged.
3. **CE text view (GPU, optional, after cfs):** score native-script band pairs a second time with the name replaced by
   its transliteration (same view on the tsim and test bands, only those rows), then average with the raw score.
   IndicXlit (MIT) gives better name transliteration than rule-based tools, but it is another model to verify and
   license-check. Only use it if the rule-based feature shows a gain and GPU time remains.

**Not tonight:** replacing unidecode inside `norm.py`. That changes every normalized field and needs the 9–12 h
rerun. Put it in Part II §II.5 "if the deadline is later".

### L2 — Content-identical twins  (CPU, 20 min, measurable)
The leaders' rule "do not keep only the topmost of content-identical records" suggests that twins can belong to the
same business.
1. **On train GT:** for S2/S3 groups with identical (name, address, country), what fraction have *all* members
   matched to the **same** S1? And what fraction have one member matched and the other unmatched?
2. If "same S1" is ≥ 90%: add a **twin-completion** step. When a selected record has a twin, (a) make the twin a
   candidate (a legitimate blocking channel: identical text) and (b) select it.
   - Check on tsim, where the earlier measurement found only 17 affected cases because twins are usually already
     both candidates.
   - Re-measure with the *current* model (fx) and **on test France**, where twins are most frequent (~2.9%).
3. If "same S1" is low: twins are deliberate distractors. Then the reverse rule helps precision: when twins compete
   for different S1s, lower both. Test on tsim.

Expected: tiny on tsim (already measured ~0), possibly non-zero on France. Cheap enough to settle.

### L3 — Per-country stage 2 for US and India  (CPU, ~1 h, measurable)
The leaders allow country as a feature (open set).
- US (0.9907) and India (0.9917) have different noise: US names are token-neutral, while India S2 adds ~2 tokens by
  transliteration; address Jaccard runs in opposite directions.
- Variants:
  1. Add `ctry_seen` as a categorical feature (US / India / other); for test France it is "other", so trees fall back
     to shared splits.
  2. Train US-only and India-only stage-2 models plus the pooled model. Use the country model for its country and the
     pooled model for any unseen country. Blend in logit space with the pooled model per country, weight tuned on
     tsim.
- **Adopt** if tsim ≥ +0.00005 with ≥ 3 of 5 folds positive. France keeps the pooled model, so there is no France
  risk.

### L4 — Language-level word frequency for the generic-word swap  (France, CPU, ~40 min, LB-only)
The leaders named **wordfreq** as allowed. The France failure mode is "a generic word swapped at the same address =
different business" versus "a generic suffix dropped/added = same business".
- Our IDF is corpus-based (measured +0.00002 via xf3).
- A *language* frequency adds a new axis: whether the differing token is an ordinary dictionary word
  (école, club, amicale, comité) or an invented / distinctive token (Iriirilum, Yumacalo).
- Feature: `wf_diff_max` / `wf_diff_min` = wordfreq zipf of the tokens that differ between the two names, in the
  record's language (en for US, fr for France; for India use en on the transliterated form). Retrain stage 2.
- tsim can only show it is harmless; the benefit is France-side, so use one LB upload. Lower priority than L1–L3.

### L5 — Compliance and reproducibility checklist  (**must do before zipping**, ~45 min)
1. **Commit the `frlab_test` generator script.** It is used by `aux=` in fx/fxa/fxc and currently exists only in an
   agent transcript.
2. Commit `fr_aug.py`, `geo.py`, `package.sh`, and the uncommitted `ce.py` / `train.py` / `decide.py` diffs
   (`aug=`, `aux=`, `ew=`, `norm1=`).
3. Model table: name, licence, parameter count, HF revision for **every** model, including e5-small (ANN), all CEs,
   and Qwen3-Reranker-0.6B/4B. Add a sentence saying no model > 8B was used at any stage, including as a teacher.
4. Data compliance paragraph that quotes the leaders' answers:
   - test-set statistics allowed;
   - self-training / synthetic pairs allowed;
   - stricter threshold for unseen countries allowed;
   - hand-written state/legal/street lists allowed;
   - no city-level or postal tables, no libpostal, no cleanco.
5. Candidate set paragraph: 103 / S1 first pass → 4.47 / S1 fed to stage 2 (total 7.74M pairs), plus the recall
   ceiling at each stage.
6. `Documentation_template.md`: no page limit, so include the loss decomposition, the France method and the
   ablation table.

### L6 — Upload policy from the leaders' rules
- Private = best public; public and private come from the same files; France is in both. So **every remaining slot
  today should be used**. A weaker upload cannot hurt, and an extra variant is an extra draw at the best public.
- Winner's curse is small because both splits are large, but prefer the variant with the better **tsim OOF** when two
  public scores are within 0.00003.
- The final zip must reproduce **exactly** the best-public upload (same `matching_results.tsv`, same candidate file).

## IV.3 What the leaders' answers rule out or allow-but-skip
- **> 8B anything** (teacher, generator, judge): prohibited. Nothing changes; we never used one.
- **Offline ≤ 8B LLM as an augmentation model**: allowed ("a preprocessing/augmentation model counts" under the
  limit). Not worth it tonight: the zero-shot LLM judge scored AUC 0.53, and a 7B generator is hours of GPU.
- **Translation models / APIs**: online ones are banned. Offline translation is allowed but irrelevant, because the
  data needs *transliteration* (same name, other script), not translation.
- **City/postal tables, libpostal, cleanco**: banned. Keep geography learned from the provided records (`geo.py`) and
  hand-written state lists only.

## IV.4 Updated tonight order (Part II + III + IV merged)
1. **N1** merge of the two machines' stacks + **fxq** → upload #1
2. **cfs + aux** + winners of **N2 / N3 / III.2-A / L3** → upload #2
3. **L1** cross-script features (if the slice gate passes) and **L2** twin rule → upload #3 (isolated India/France
   deltas)
4. **N5** per-source France δ or **L4** wordfreq → upload #4; keep one slot in reserve for a rollback
5. **L5** checklist in parallel from 20:00; freeze at 21:00; zip = the best public upload

Expected contribution of Part IV: **L1 +0.0000–0.0003 LB** (India is 47% of test, so an India-only +0.0005 OOF is
worth about +0.00025 LB), **L3 +0.00002–0.0001**, **L2 ~0 to +0.0001**. Its biggest value is **L5**, because the top
packages are audited before ranks are confirmed.
