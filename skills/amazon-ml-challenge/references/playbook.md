# 72-hour playbook, rules, write-up and pitfalls

## Format facts (2025; re-check the new rules on day 1)
- 72 hours, teams of 3 to 4 from one institute. Public LB on a subset (25K of 75K test rows in 2025).
- Final rank used the full test set AND the 1 to 2 page approach document. Top 10 present to Amazon
  scientists; the Top 50 in 2025 were eligible for Applied Scientist intern interviews.
- 2025 rules: no external price lookup or scraping; models MIT or Apache 2.0 licensed and at most 8B
  parameters (as reported by a Rank 80 team). Verify every model's licence and size before downloading.
- Submissions may be capped (a 2024 writeup mentions 15). Budget them.

## Timeline for 4 people

| Hours | Lead | Text / LLM | Vision / VLM | Tabular / infra |
|---|---|---|---|---|
| 0 to 3 | rules + metric geometry; folds.csv | text EDA, parse fields | start download_images.py | GPUs, tracker, sanity.py on sample sub |
| 3 to 12 | baseline submission (gbdt_cv + blend); worst-OOF review | fine-tune text encoder | frozen image embeddings; concat model | LLM field extraction; knn_feats |
| 12 to 30 | curation; adv_val | encoder swaps; metric loss | fine-tune vision; resolution | Optuna tuning; target encoding |
| 30 to 50 | ablation log; pick 4 to 6 diverse members | fusion upgrade | aux losses; seeds | per-unit model; stacking |
| 50 to 62 | blend weights, calibration | full-data retrains | full-data retrains | sharded test inference |
| 62 to 72 | write-up | reproducibility run | reproducibility run | final sanity + early upload |

Rules of thumb: valid submission in the first 6 hours; freeze new ideas at hour 50; select the final
submission by OOF; keep a safe backup blend.

## Experiment log (keep from hour 1; it becomes the write-up's ablation table)

```
exp_id | owner | change (one lever) | OOF | fold std | LB | kept? | notes
```

## Approach document template (1 to 2 pages)

1. Problem and metric; how the metric shaped our loss, blending and post-processing.
2. Data insights from EDA (skew, noise, duplicates, missing values) and the cleaning we did.
3. Features (parsed quantities, encodings, embeddings, kNN) with their measured OOF gain.
4. Models (architecture, loss, key hyperparameters) and validation scheme (folds, grouping).
5. Ensemble and calibration (weights, multiplier / threshold) with OOF before and after.
6. Ablation table (from the log). 7. Compliance: no external data; each model's licence and parameter count.
Be ready to defend every choice: the 2025 Rank 20 author was later questioned on why each model,
why LoRA, why AdamW.

## Pitfalls that cost ranks (all observed or classic)

- Tuning on the public LB (only a third of test in 2025). Trust OOF.
- Target encoding or kNN features computed on the full train (silent leak; OOF looks amazing, LB does not).
- Near-duplicate rows split across folds (CV lies). Use group folds.
- Running out of GPU time for test inference (Rank 357, 2024 finished ~35%).
- Subsampling huge data for convenience (2021: ~12 points lost).
- Fine-tuning on large noisy labels instead of a clean small set (2024: 150K SFT < zero-shot).
- Dropping test rows with missing inputs; forgetting missing-image rows.
- Wrong output format (units, decimals, empty-string convention) wasting a submission.
- Notebook hidden state: rerun the final pipeline end to end from raw files before the last upload.
- Using a model with a disallowed licence or over the parameter cap.
