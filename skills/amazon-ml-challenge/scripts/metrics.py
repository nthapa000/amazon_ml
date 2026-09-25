"""Competition metrics, calibration helpers and metric-aligned losses.

Usage from other scripts:  from metrics import M, tune_mult, tune_thr
M[name] -> (fn(y, p), 'min'|'max', kind)  where kind is 'reg' or 'clf'
"""
import numpy as np


def smape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    d = np.abs(y) + np.abs(p)
    return 100 * np.mean(np.where(d == 0, 0, 2 * np.abs(p - y) / np.where(d == 0, 1, d)))


def mape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return 100 * np.mean(np.abs(p - y) / np.abs(y))


def mape_score(y, p):          # Amazon ML 2023: max(0, 100*(1-MAPE))
    return max(0.0, 100 - mape(y, p))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def rmsle(y, p):
    return rmse(np.log1p(y), np.log1p(np.clip(p, 0, None)))


def acc(y, p):
    return float(np.mean(np.asarray(y) == np.asarray(p)))


def f1(y, p):
    y, p = np.asarray(y).astype(int), np.asarray(p).astype(int)
    tp = np.sum((y == 1) & (p == 1)); fp = np.sum((y == 0) & (p == 1)); fn = np.sum((y == 1) & (p == 0))
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def macro_f1(y, p):
    from sklearn.metrics import f1_score
    return float(f1_score(y, p, average='macro'))


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


M = {
    'smape': (smape, 'min', 'reg'), 'mape': (mape, 'min', 'reg'), 'mape_score': (mape_score, 'max', 'reg'),
    'rmse': (rmse, 'min', 'reg'), 'mae': (mae, 'min', 'reg'), 'rmsle': (rmsle, 'min', 'reg'),
    'acc': (acc, 'max', 'clf'), 'f1': (f1, 'max', 'clf'), 'macro_f1': (macro_f1, 'max', 'clf'),
    'auc': (auc, 'max', 'clf'),
}


def better(a, b, mode):
    return a < b if mode == 'min' else a > b


def tune_mult(y, p, fn, mode, lo=0.8, hi=1.2, n=81):
    """Global multiplier on predictions (regression). Tune on OOF only."""
    best_k, best = 1.0, fn(y, p)
    for k in np.linspace(lo, hi, n):
        s = fn(y, k * p)
        if better(s, best, mode):
            best_k, best = k, s
    return best_k, best


def tune_thr(y, prob, fn, mode, n=97):
    """Decision threshold for binary probabilities (F1, acc...). Tune on OOF only."""
    best_t, best = 0.5, fn(y, (prob > 0.5).astype(int))
    for t in np.linspace(0.02, 0.98, n):
        s = fn(y, (prob > t).astype(int))
        if better(s, best, mode):
            best_t, best = t, s
    return best_t, best


def snap(p, grid):
    """Round each prediction to the nearest value that appears in the training target."""
    g = np.sort(np.unique(grid))
    i = np.clip(np.searchsorted(g, p), 1, len(g) - 1)
    lo, hi = g[i - 1], g[i]
    return np.where(np.abs(p - lo) <= np.abs(hi - p), lo, hi)


TORCH_LOSSES = '''
# paste into a PyTorch training loop; p, y are log(target) tensors
import torch
def smape_log(p, y):                      # exact SMAPE (0..2) since SMAPE = 2*tanh(|ln k|/2)
    return (2 * torch.tanh((p - y).abs() / 2)).mean()
def mape_log(p, y):                       # exact MAPE in log space: |e^(p-y) - 1|
    return (torch.exp(p - y) - 1).abs().mean()
# warm start: L1 on log target for 2-3 epochs, then 0.5*L1 + smape_log
'''

if __name__ == '__main__':
    y = np.array([100, 50, 10.0]); p = np.array([120, 40, 10.0])
    print('smape', smape(y, p), 'mape_score', mape_score(y, p))
    print(TORCH_LOSSES)
