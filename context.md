# Session context handoff (written 2026-09-27 ~17:30 IST, status updated 18:58)

Previous chat was monitoring only: no code was changed. It read the docs, watched the running GPU job,
estimated its progress, and modelled the expected leaderboard score.

## Project in one paragraph
Amazon ML Challenge 2026, Business Entity Resolution. The task is to link Source-1 (S1) businesses to
Source-2/3 records by name, address and country. The metric is macro F0.5 per S1. The test set is
India 47%, US 38% and France 15%; France never appears in train and is the whole gap to rank 1.
**Deadline: 2026-09-27 23:59 IST.**

Read these first:
- `er/PLAN.md`: status log at the top and bottom
- `er/EXPERIMENTS.md`: ablation log
- `er/VERSIONS.md`: released tags and scores
- `er/README.md`: the 22-step pipeline

`approach.md` and `CHANGES_kushal_local.md` were written by the teammate's parallel session (account
m25csa011, branch `kushal/local`).

**Rules from memory:**
- Coordinate before editing `er/src/`, because peer sessions share it. Put new probes in `er/ana/`.
- Every real release writes both `matching_results.tsv` and `candidate_pairs.tsv`.

## Scores so far
- Best released but not yet uploaded: `b34c_frm2`, OOF 0.991082 (tsim, US+India only).
- Team's best public LB: 0.987449 (`cx1`, teammate). Our `probe_fin_frm2` scored 0.986946. Rank 1 is 0.990788.
- Some notes quote 0.987862 (Kushal) as the team best public score. Verify with the user.

## Running jobs (updated 18:58; SLURM user m25csa019, partition mtech: max 2 running / 3 submitted / 2 GPUs)

| job | name | state | what it does |
|---|---|---|---|
| 41397 | `fz_ceq` | **DONE 18:47**, merged band AUC **0.98563** (both halves), `ceq_b1ka_{train,test}.parquet` | Qwen3-Reranker-4B LoRA CE |
| 41942 | `fz_ceqf` | RUNNING cn07 since 18:55 (GPU IDX 1-2 Blackwell, **-c 4 --mem=140G**), ETA ~22:00 | French continuation of ceq; re-scores French test pairs; merge into ceq |
| 41941 | `fz_f5` | RUNNING cn07 since 18:55 (24 CPU, mem lowered to 60G), ETA ~20:05 | fin5a + fin5b (with ceq) -> fin5 -> fin5c, **fin5c_frm2** |
| 41943 | `q_f5x` | PENDING afterok:41941 (24 CPU, 60G) | Q1 France diag on fin5c; Q5 fin5sa (sup3) -> fin5s -> fin5sc + fold check; Q3 blend fin5+b34 -> b534c + fold check |

**Why ceqf was restarted (18:52):** f5 (24 CPU) could not fit any node (cn07 had 12 CPU / 27 GB left after the old
ceqf 41515 took 16 CPU / 200 GB and the friend's `fr_final` took 16 CPU / 56 GB), and the scheduler ETA was 01:41
tomorrow. 41515 was cancelled ~5 min in and resubmitted as 41942 with identical ARGS/MERGE but 4 CPU / 140 GB.
`bw_retry.sh` for ceqf has exited, so there is **no automatic retry** for 41942. Stage-2 trainings peak at 18-22 GB
(sacct MaxRSS of fz_f3 / fz_f4), which is why f5 fits in 60 GB. `jobs/par.sbatch` header is now 90G.

**Quick fixes from `plan (1).md`** (new `er/ana/qfix.py`, decision layer only, no `er/src/` edits; see EXPERIMENTS.md):
- Q2 count prior + per-source caps on b34c: all within +-0.000002 -> not adopted.
- Q4 per-source France shift: dropped (S2 and S3 under-linked by the same proportion).
- Q1: France empty rate matches US/India at fr ~ -1.2 on b34c (current -2) -> one France probe upload later.
- France is under-linked at every fr (partial recall inside matched S1) -> model-side, not calibration.
- Q5 input built: `sup3_ka_{tsim,test}` (sibling support from fin3 stage-2 scores, same row set as `sup_ka`).
- Gates: tsim >= +0.00005 and >= 3/5 folds (ana/cmp_folds.py). Each adopted fix gets its own snap.py version.
- Friend merge (N1) is last priority; the list of files to request is in the plan file
  `~/.claude/plans/checkout-plan-1-md-and-see-lazy-crown.md`.

## What runs next (defined in `er/jobs/fin56.dag`; runner log `logs/dag_fin56.log`)
ceq finished 18:47 and f5/ceqf started 18:55, so the times below hold (+~10 min).

**Branch A, CPU (~1h10m after ceq, ready ~20:00)**
1. `fz_f5` trains `fin5a` (fin3 features + ceq) and `fin5b` (fin4/cef + ceq) in parallel.
2. It blends them into `fin5` and runs a fold check against `b34`.
3. It applies `cands.py` thr=0.005 and runs the validator.
4. It releases `fin5c` and **`fin5c_frm2`** (France shift fr=-2). `fin5c_frm2` is the safe upload.

**Branch B, GPU (ready ~22:30)**
1. `ceqf` runs for about 3 h.
2. `fz_w6` waits for `ceqf_b1ka_test.parquet`.
3. `fz_f6` re-scores the test set with fin5's models and ceqf patched in, with no retraining.
4. It releases **`fin6c_frm2`**. `ana/diff_country.py` must show that only French rows changed.

Outputs land in `/scratch/m25csa019/er_out/<tag>/`.

## Status-check commands
```
squeue -u m25csa019 -o "%.10i %.20j %.8T %.10M %R"
tail -n 3 er/logs/fz_ceq_41397_h0.log er/logs/fz_ceq_41397_h1.log
srun --jobid=41397 --overlap nvidia-smi          # direct ssh to cn07 is denied; this works
ls -la /scratch/m25csa019/er_final/ceq*.parquet   # appear after each half finishes, then merged file
tail er/logs/dag_fin56.log
```

## Score prediction model (use once fin5/fin6 OOF is printed by `decide.py tune`)
Calibrated from the `probe_c6_noFR` probe and every observed (OOF, LB) pair; it is consistent across all tags:

```
LB ≈ 0.8503 × (OOF_tsim + 0.00054) + 0.1497 × F_France
```

France F implied by each tag's LB:

| tag | c2 | c6 | c5/fin | fin fr=-2 | cx1 |
|---|---|---|---|---|---|
| France F | 0.946 | 0.955 | 0.960 | 0.9615 | 0.9639 |

France cannot be observed offline, because tsim has no France. It is the uncertain 15% of the score, and each reranker upgrade has moved France 3–10× more than it moved OOF.

**Expected with ceq:**
- OOF about 0.9913–0.9916.
- France about 0.967–0.972.
- LB **≈0.988–0.989** (central estimate ~0.9885).
- `ceqf` may add another +0.001–0.002.
- Rank 1 would need France ≈0.985.

Public→private sampling noise is about ±0.0001–0.0002. The private ranking uses the best *public* submission.

The calculation script is in this session's scratchpad (`lbpred.py`). To recreate it, apply the formula above.

## Pending with the user
- Once `ceq`/`fin5` finish, compute the expected LB from the actual fin5 OOF (and later fin6's).
- Decide the final upload: `fin5c_frm2` (~20:00) versus `fin6c_frm2` (~22:30).
