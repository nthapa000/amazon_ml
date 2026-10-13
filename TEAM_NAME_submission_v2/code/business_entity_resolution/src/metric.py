"""Macro F_beta exactly as the README defines it.

Per S1 entity:
  - truth empty & pred empty  -> 1.0
  - truth empty & pred nonempty -> 0.0
  - truth nonempty & pred empty -> 0.0
  - else F_beta from precision / recall of the id sets
Then mean over ALL S1 entities in the evaluation set.
"""
import numpy as np


def fbeta_one(t, p, b=0.5):
    """F-beta of one S1 entity from its true and predicted id sets (empty vs empty scores 1)."""
    if not t:
        return 1.0 if not p else 0.0
    if not p:
        return 0.0
    tp = len(t & p)
    if tp == 0:
        return 0.0
    pr, rc = tp / len(p), tp / len(t)
    b2 = b * b
    return (1 + b2) * pr * rc / (b2 * pr + rc)


def macro_fbeta(truth, pred, b=0.5, ids=None):
    """truth, pred: dict s1_id -> set of ids. ids: evaluation S1 ids (default: truth keys)."""
    ids = list(truth.keys()) if ids is None else list(ids)
    return float(np.mean([fbeta_one(truth.get(i, set()), pred.get(i, set()), b) for i in ids]))


def macro_fbeta_arr(n_true, n_pred, n_tp, b=0.5):
    """Vectorised version from per-entity counts (numpy arrays)."""
    n_true, n_pred, n_tp = (np.asarray(x, dtype=np.float64) for x in (n_true, n_pred, n_tp))
    b2 = b * b
    f = np.where(n_tp > 0, (1 + b2) * n_tp / (b2 * n_true + n_pred + 1e-12), 0.0)
    f = np.where((n_true == 0) & (n_pred == 0), 1.0, f)
    return float(f.mean()), f


def read_idlists(path):
    """Read a ground-truth / submission TSV into dict s1 -> set."""
    out = {}
    with open(path, encoding='utf-8') as fh:
        next(fh)
        for ln in fh:
            s1, _, rest = ln.rstrip('\n').partition('\t')
            out[s1] = set(rest.split(',')) if rest.strip() else set()
    return out


if __name__ == '__main__':
    t = {'S1-00001': {'S2-00047', 'S3-00812'}, 'S1-2': set(), 'S1-3': set(), 'S1-4': {'a'}}
    p = {'S1-00001': {'S2-00047', 'S2-00193', 'S3-00812'}, 'S1-2': set(), 'S1-3': {'x'}, 'S1-4': set()}
    f1 = fbeta_one(t['S1-00001'], p['S1-00001'])
    assert abs(f1 - 0.714) < 1e-3, f1
    m = macro_fbeta(t, p)
    assert abs(m - (f1 + 1 + 0 + 0) / 4) < 1e-9
    # vectorised agrees; F = (1+b2)tp/(b2*T + P)
    mv, _ = macro_fbeta_arr([2, 0, 0, 1], [3, 0, 1, 0], [2, 0, 0, 0])
    assert abs(mv - m) < 1e-9, (mv, m)
    print('metric ok: README example =', round(f1, 3), 'macro =', round(m, 4))
