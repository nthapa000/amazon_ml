# Metric engineering (read first, whatever the problem)

Every Amazon ML Challenge so far used a metric that is NOT MSE: accuracy (2021), 100*(1-MAPE) (2023),
exact-match F1 (2024), SMAPE (2025). Optimising what is actually scored was a free win every year.

## Contents
1. Ratio-space geometry of SMAPE and MAPE
2. Per-metric playbook (loss, blend, post-processing)
3. Classification: thresholds, confidence, abstention
4. Code snippets

## 1. Ratio-space geometry

Write k = pred / actual.

- SMAPE = 2|k-1| / (1+k) = 2 * tanh(|ln k| / 2)
  - symmetric in log ratio: predicting 2x and 0.5x cost the same (66.7%)
  - saturates at 200%: far misses stop mattering, so it behaves like a robust loss in log space
- MAPE = |k-1|
  - under-prediction capped at 100%, over-prediction unbounded
  - the optimal constant prediction is the 1/y-weighted median, below the ordinary median: shrink predictions down

| k | 0.2 | 0.5 | 0.8 | 1.25 | 2 | 5 |
|---|---|---|---|---|---|---|
| SMAPE % | 133.3 | 66.7 | 22.2 | 22.2 | 66.7 | 133.3 |
| MAPE % | 80 | 50 | 20 | 25 | 100 | 400 |

## 2. Per-metric playbook

| Metric | Train on | Loss | Blend | Post-process | Evidence |
|---|---|---|---|---|---|
| SMAPE | log(y) | L1 or Huber warm-up, then exact SMAPE (2*tanh) | geometric mean (log space) | OOF multiplier, clip to train range, snap if values repeat | Rank 8 (Huber on log2), Rank 20 (direct SMAPE), 2025 |
| MAPE / 100*(1-MAPE) | log(y), clip log target | L1 in log space, or exact MAPE in log space | take minimum or shift blend down | multiplier usually < 1, snap to train grid | Rank 2, 2023 (min of BERT and RoBERTa, snap) |
| RMSE | y | L2 | arithmetic mean | clip | standard |
| RMSLE | log1p(y) | L2 on log1p | mean in log1p space | clip at 0 | standard |
| MAE | y | L1 | median or mean | none | standard |
| Accuracy (many classes) | labels | cross-entropy, label smoothing 0.1 | average probabilities / rank-average | prior correction if class mix shifts | 2021 teams |
| Binary F1 / precision / recall | labels | logloss (class weights if imbalanced) | average probabilities | tune the decision threshold on OOF | scientist briefing; sklearn docs |
| Exact-match F1 on strings (2024) | text generation | SFT cross-entropy | majority vote on normalised strings | grammar normaliser, allowed-unit filter, abstain below confidence | Ranks 1, 6, 45, 2024 |

Why exact SMAPE needs a warm start: tanh saturates, so the gradient vanishes for very wrong
predictions. Start with L1 on log target for 2 or 3 epochs, then add the SMAPE term.

## 3. Classification: thresholds, confidence, abstention

- A model outputs a probability. 0.93 = confident; about 0.5 = confused. The Amazon scientist stressed that
  confidence matters a lot in the challenge.
- 0.5 is only a default. scikit-learn's docs show a case where the best threshold was about 0.02. In the
  synthetic churn test of this skill, tuning moved OOF F1 from 0.438 to 0.578.
- Always tune the threshold on OOF predictions, never on the test or the public leaderboard.
- Precision = how right your positive predictions are. Recall = how many real positives you catch.
  F1 = harmonic mean. Decide which one the metric rewards before modelling.
- Exact-match or extraction metrics: check how the grader scores an empty output versus a wrong one.
  If a wrong answer costs more (e.g. counts as a false positive), emit empty below a tuned confidence.
- Calibration: if 0.9 predictions are right only 70% of the time, tune thresholds on calibrated scores
  (isotonic on OOF) or blend in log-odds space.

## 4. Snippets

```python
from metrics import M, tune_mult, tune_thr, snap      # scripts/metrics.py
fn, mode, kind = M['smape']
k, s = tune_mult(y, oof, fn, mode)                    # global multiplier
t, s = tune_thr(y, oof_prob, M['f1'][0], 'max')       # decision threshold

# PyTorch, p and y are log-target tensors
def smape_log(p, y): return (2 * torch.tanh((p - y).abs() / 2)).mean()
def mape_log(p, y):  return (torch.exp(p - y) - 1).abs().mean()
```

A metric not listed here: write it into `M` in scripts/metrics.py first, then derive its geometry the
same way (express it as a function of pred/actual or of the confusion matrix) before choosing a loss.
