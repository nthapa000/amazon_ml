# plan.md — closing the gap to the top 3 (Amazon ML Challenge 2026, Business Entity Resolution)

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
