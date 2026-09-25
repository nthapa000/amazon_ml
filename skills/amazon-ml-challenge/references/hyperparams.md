# Hyperparameters: defaults, search spaces, tuning protocol

## Protocol (fast and honest)

1. Fix folds first (make_folds.py). Tuning on a different split than evaluation is how LB overfitting starts.
2. Baseline with defaults (gbdt_cv.py, lr=0.03, early stopping 300). Record OOF.
3. Tune with a raised learning rate (lr=0.08) and a small budget (trials=20 to 40) to find structure params.
4. Refit the best structure at a low learning rate (lr=0.02 to 0.03); early stopping picks the rounds.
5. Keep a tuned config only if it beats the baseline by more than fold-to-fold noise
   (rule of thumb: > 0.5 x std of fold scores).
6. Final models: 3 to 5 seeds averaged (log space for regression).

gbdt_cv.py does steps 2 to 4: search with `trials=30 lr=0.08 tag=t1`, then refit with
`params=params_t1.json lr=0.03 tag=t1f` (reuses the tuned structure at a lower learning rate).

## LightGBM (gbdt_cv.py model=lgb)

| Param | Default here | Search range | Notes |
|---|---|---|---|
| learning_rate | 0.03 | fixed while tuning | lower for final fit |
| num_leaves | 63 | 15 to 255 (log) | main capacity knob |
| min_data_in_leaf | 20 | 5 to 200 (log) | raise for noisy targets |
| feature_fraction | 0.8 | 0.4 to 1.0 | lower with many SVD / kNN feats |
| bagging_fraction / freq | 0.8 / 1 | 0.5 to 1.0 | |
| lambda_l1 / l2 | 0 / 1 | 1e-3 to 10 (log) | |
| objective | l1 on log for SMAPE/MAPE | l1, huber, l2 | match the metric |

## XGBoost (model=xgb) — also the SageMaker built-in algorithm

| Param | Default here | Search range | SageMaker name |
|---|---|---|---|
| eta | 0.03 | fixed while tuning | eta |
| max_depth | 8 | 3 to 12 | max_depth |
| min_child_weight | 5 | 1 to 50 (log) | min_child_weight |
| subsample | 0.8 | 0.5 to 1.0 | subsample |
| colsample_bytree | 0.8 | 0.4 to 1.0 | colsample_bytree |
| reg_alpha / reg_lambda | 0 / 1 | 1e-3 to 10 (log) | alpha / lambda |
| rounds | early stopping | cap 20000 | num_round (use early_stopping_rounds) |
| objective | reg:absoluteerror on log for SMAPE; binary:logistic for churn-like | | objective |
| scale_pos_weight | 1 | neg/pos ratio for imbalance | scale_pos_weight |

The session's demo used num_round=100 and objective=binary:logistic; in competition, replace a fixed
round count with a large cap plus early stopping on validation.

## CatBoost (model=cat)

depth 8 (4 to 10), l2_leaf_reg 3 (1 to 30 log), random_strength (0.01 to 10 log), lr 0.06, od_wait 300.
Strong with categorical columns and target-encoded features; often the best single GBDT on catalog data
(it was in this skill's synthetic test).

## Transformer / multimodal fine-tuning (Rank 8 and Rank 20, 2025)

| Knob | Reported setting |
|---|---|
| Optimiser | AdamW, cosine schedule, 6% warmup |
| Encoder LR | 2e-5 (head 10x to 50x higher) |
| Weight decay | 0.01 |
| Grad clip | 1.0 |
| Precision | FP16 / BF16 mixed precision, gradient accumulation |
| Batch | 16 to 32 effective via accumulation |
| Epochs | 15 beat 5 to 10 for CLIP + DistilBERT |
| Loss | Huber on log2(price) + 0.2 image-text InfoNCE + 0.1 SimCSE InfoNCE [Rank 8]; direct SMAPE [Rank 20] |
| Head | 512-256-128 FFN, BatchNorm, GELU (deeper overfits) [Rank 20] |
| Two-stage | validate to find epochs, retrain on full data [Rank 8] |
Extras [General]: layer-wise LR decay, EMA/SWA of weights, 3 to 5 seeds.

## QLoRA for VLM / LLM (Ranks 1 and 6, 2024)

| Knob | Setting |
|---|---|
| Tool | LLaMA-Factory (or Unsloth) |
| Quantisation | 8-bit (Rank 1, Rank 6) or 4-bit to fit smaller GPUs |
| LR | 5e-5; ReduceLROnPlateau on small curated sets (Rank 1) |
| Batch / grad accumulation | 8 / 8 |
| Epochs | 1 on 150K rows (Rank 6) vs several on 1.6K curated rows (Rank 1, the winner) |
| LoRA rank / alpha [General] | r 16 to 64, alpha = 2r, dropout 0.05, target all linear layers |
| Images [Rank 45] | resize > 980 px; rotation augmentation |

## SageMaker Automatic Model Tuning (if Amazon provides SageMaker)

Give ranges (e.g. eta 0.01 to 0.3, max_depth 3 to 10, min_child_weight 1 to 20, alpha 0 to 5) and the
objective metric (validation:rmse, validation:auc, validation:f1...). Bayesian search, run jobs in
parallel. See references/sagemaker.md.
