# Approach: raising macro F0.5 (Amazon ML Challenge 2026 — Business Entity Resolution)

## 1. Context

Link every Source-1 (S1) business record to its Source-2/S3 matches using only name, address and
country. Metric: macro F0.5 per S1 entity (a true singleton predicted empty scores 1; every S2/S3
id belongs to at most one S1). Test set: 1.73M S1 — India 47%, US 38%, **France 15% (never in
train)**.

Two codebases exist:
- **kushal** = current code, branch `kushal/local`, `/scratch/m25csa011/from_kushal/amazon_ml/er`
  (mirrored at `mahek/Amazon/amazon_ml/er`). Portability refactor (`paths.py`, local DAG runner) —
  no pipeline-logic changes.
- **main** = the zip, teammate m25csa019's newer branch (`mahek/Amazon/amazon_ml-main.zip`). Adds
  `feats3.py`, `fradapt.py` (French self-training), `llm.py` (dropped), `cands.py` (final candidate
  pruning), extra `ana/*` tools, and `cer2`/`cef`/`ceq` cross-encoder runs.

**Deadline 2026-09-27 23:59 IST.** 5 uploads/day, teammate m25csa019 uploads. Teammate is currently
running `ceq` (Qwen3-Reranker-4B LoRA CE) and has run `cef` (French self-trained CE) on their own
account (`/scratch/m25csa019`, not readable by this user).

## 2. Current standing

| tag | branch | OOF (tsim) | public LB |
|---|---|---|---|
| `cx1` = fin + `cx`(bge-reranker-v2-m3) + `fr=-2` | kushal | 0.99106 | **0.987449 (best)** |
| `fin3` = fin2(Optuna) + `cer2`(bge-reranker-v2-m3) | main | 0.991115 | not yet uploaded |

Rank 1 on the leaderboard: **0.990788**. LB ≈ 0.8503·(US+India F) + 0.1497·(France F).
LB-implied: **France ≈ 0.966**, **US+India ≈ 0.991**.

Stage-2 gain shares (cx1): `c_marg2` 0.77, `c_p` 0.13, `cx` 0.053 — everything else combined is
under 0.03. The model is essentially "stage-1 competition margin + one strong cross-encoder".

## 3. Is LB 0.995 reachable? — No, and here is the arithmetic

- Perfect matcher restricted to the **already-scored candidates** on US+India = **0.99666**
  (measured ceiling, `CHANGES_kushal_local.md` §7).
- 0.85·0.99666 + 0.15·F_France = 0.995 ⟹ France would need to be **≈1.015** — impossible.
- Even with a flawless matcher on US+India, France would need **0.9856** (up from ≈0.966).
- Rank 1 (0.990788) implies their France is ≈0.986 — so **the entire gap to rank 1 is France**,
  not model quality on US+India.
- **Realistic target for today: LB ≈ 0.988–0.990.** Beating rank 1 needs France to gain ≈0.02 of
  its own F-score, which is a lot to ask from any one lever in the time left, but the two paths
  below (a better reranker, French self-training with a stronger teacher) are the ones the
  project's own history says pay off.

Every lever below is judged against this 0.988–0.990 target, not against 0.995.

## 4. Where the remaining loss is

- **US+India:** current OOF 0.9911 vs ceiling 0.99666 → **0.0056 of headroom**, of which blocking
  recall is worth at most ~0.0005 more (residual misses are mostly inherently ambiguous: common
  names with empty/city-only addresses). **~90% of US+India headroom is model quality, not
  recall.**
- **France:** no labels exist. LB decomposition (`CHANGES_kushal_local.md` §5, §7) shows every past
  gain in raw-text multilingual reranker quality moved France disproportionately (`ceb` +0.00014
  OOF → +0.0015 LB, ~10×; `cx` +0.00016 OOF → +0.000503 LB, ~3×). France's known failure mode:
  distractors **substitute** one generic word at the same address ("Nantes College" vs "Nantes
  Amicale"); true matches **drop** words ("Green Creative Dynamics" → "Green Creative"). Aggregate
  similarity features can't separate these.

## 5. Measured dead ends — do not repeat

Decision-rule grid (saturated), `xf3`/`xf2`/`feats3` hand-engineered features (each +0.00000 to
+0.00007), entity gate, XGBoost blend, Optuna (+0.000014), seeds (noise 0.000004), LLM zero-shot
judge (AUC 0.53), a second reranker of the *same* family as `cx`/`ceb` (corr 0.984, no independent
signal), expansion round 2 (0.6% recall), K200 caps.

## 6. Ranked levers (by expected LB gain ÷ hour, judged against the 0.988–0.990 target)

1. **Merge branches onto the strongest reranker + Optuna params, then add `cx`/`cer2` together only
   if independent.** `cer2` (main, OOF+0.000159 over `ceb`, adopted as `fin3`) and `cx` (kushal, band
   AUC 0.9835) are the *same base model* (bge-reranker-v2-m3) trained separately — likely redundant
   like `cx` vs `ceb` was. Test correlation first (cheap, polars only); if AUC of the residual after
   regressing one out of the other is near 0.5, keep only the stronger one, freeing a GPU-hour
   budget for something architecturally different.
   - *Needs:* `fin3`/`cer2`/Optuna-param artifacts copied from `/scratch/m25csa019` (arranging this).
2. **Finish/adopt `ceq` (Qwen3-Reranker-4B LoRA) if it lands in time.** Architecturally different
   from the encoder-family rerankers (`cx`/`cer2`/`ceb`), which is exactly the kind of diversity that
   worked before (three-CE stack). Risk: timing estimate from `logs/fz_ceq_*` puts its ETA around
   15:00–17:00 IST, uncomfortably close to the team's 20:00 freeze. **Coordinate with the teammate
   before committing to it** — if it's not done and scored by ~18:00, don't wait on it for the final
   submission.
3. **Re-run the French self-training set (`fradapt.py`) with a stronger teacher and a tighter
   hard-negative filter, then fine-tune the *strongest* reranker (`cx`/`cer2`) on it — not `ceb`.**
   This is the biggest single lever left, and the only one the team hasn't tried with its best
   model:
   - Current `cef` warm-starts from `ceb_b1ka` (weaker model) with teacher `fin2`. Rebuild with
     teacher `cx1` or `fin3` (stronger, more calibrated probabilities → cleaner pseudo-labels).
   - **Bug found in `fradapt.py`:** the "look-alike" hard-negative filter (`max(token_set_ratio) ≥
     80`) lets ~72% of the negative pool through unfiltered (a same-city, different-everything-else
     pair passes trivially). This makes the negative pairs too easy — they don't teach the model the
     actual France confusion (same address, one word swapped). **Fix:** require *both* name and
     address token_set_ratio to be high but not identical, or explicitly construct the "one generic
     word differs" pattern as its own negative class before sampling.
   - Warm-start from the strongest existing checkpoint (`cx_b1ka_h0/h1`, locally available) rather
     than `ceb_b1ka`.
   - This is standard-config-sized: ~600k replay + ~628k French rows, 1 epoch, same order of GPU time
     as `cef` already took (a few hours), and it directly targets the country responsible for the
     whole scoring gap. It is **organizer-permitted** (pseudo-labels on test now explicitly allowed).
4. **Missing-feature pass on the existing `s2mat` (cheap, polars-only, ~20 min per retrain, no GPU):**
   the feature audit found these are absent and computable from data already available locally:
   - **Competition computed with the cross-encoder score, not just stage-1 p** (e.g. `cx`/`cer2`
     margin over the runner-up S1, analogous to `c_marg2` which is already 77% of stage-2's gain).
   - **Contrast-against-competitor features:** for pair `(s1, x)` and its runner-up `(a, x)` (already
     joinable via `s2mat`'s `c_oth2`), the *difference* in `co_ts`, `ad_ts`, `hn_conf`, and CE scores
     between the two — a pure self-join, no rapidfuzz needed.
   - Per-source (S2 vs S3) collective count features, currently missing.
   Each is one extra column plus the existing 22-minute `train.py` retrain loop — cheap enough to try
   several today. Expected +0.0001–0.0003 OOF each; not transformative alone, but free.
5. **`decide.py fr=` re-tuning for whatever reranker mix wins.** The `fr=-2` shift was fit to `fin`'s
   score distribution; a changed model changes the optimum. Cheap (a few `decide.py apply
   fr=<delta>` probes on the LB, coordinate uploads with the teammate).
6. **Do NOT touch blocking/ANN today.** Quantified cost (rebuild ANN, `ka` union, re-score every CE
   on the new band, full stage-2 retrain) is ~10–14h of GPU-competing work for an estimated
   **+0.00005 to +0.0002 OOF** (≈+0.0001–0.0003 LB). Not worth it against the France levers above.
7. **Normalization gaps** (French "frs"→"frères", "apt", US state/street-abbreviation collisions like
   CT/MT/NE/WY/FL) — audited at ~+0.00007 LB combined, needs a 9–12h full rerun. **Skip for today.**

## 7. Bugs found worth a one-line fix (low cost, do opportunistically)

- `HANDOVER_FOR_TEAMMATE.md` §4 repro command uses `cand=cand_test_k100` (the raw ~87/S1 key set),
  not `cand_test_final`/`cands.py`'s pruned ~4.5/S1 set — **any real submission must use the pruned
  candidate set** (organizer rule: smaller `candidate_pairs.tsv` ranks higher, and it must be a
  superset of your matches).
- `fradapt.py`'s hard-negative filter (see lever 3).
- `ana/ana_miss.py` import path is broken in `main` (fixed in `kushal`).

## 8. Environment status

- Conda env **`amazon`** (`/csehome/m25csa011/miniconda3/envs/amazon`) has polars, pyarrow,
  lightgbm, xgboost, torch 2.14+cu130, transformers, rapidfuzz, optuna, sklearn — everything the
  core pipeline needs.
- Missing: **`peft`** (only for LoRA/`ceq`) — `pip install peft`.
- Torch is a **cu130** build — verify on both A30 (cn02–05) and cn07 (A6000 / Blackwell) with a
  1-minute smoke job before relying on it; fall back to the cu128 wheel if the driver's too old.
- **`/csehome` hit 100% full** (0 bytes free) earlier this session; freed since (now ~1.5 GB free on
  a 7.5 TB shared filesystem). Nothing in the training pipeline should write to `/csehome` — all
  work happens under `/scratch/m25csa011` (~71 GB free of 200 GB quota).
- `main`'s scripts hardcode paths to `/scratch/m25csa019/...`; must export `ER_WORK`,
  `ER_MODELS` etc. pointing at this user's `/scratch/m25csa011/from_kushal/{work,models}` (kushal's
  `paths.py` already does this correctly; main's files need the env var overrides).

## 9. Open items

- Confirm the teammate has copied `fin3`/`cer2` OOF+test parquets, the French `cef` output, `ceq`
  progress/checkpoints, and `params_fin.json` (Optuna) into a location this account can read.
- Coordination on submission slots (5/day) so we don't waste one duplicating the teammate's upload.
- A go/no-go check-in around 15:00–17:00 IST on whether `ceq` finished in time to include.

## 10. Verification before any submission

- `decide.py tune` OOF must reproduce the reference numbers above to 5 decimals before trusting any
  new measurement (baseline-parity gate, as `CHANGES_kushal_local.md` already established).
- `student_resource/utils/validate_submission.py --check-ids` must print `PASS`.
- Confirm `candidate_pairs.tsv` is the pruned cascade set (`cands.py thr=0.005` or equivalent), not
  the raw key-blocking set.
