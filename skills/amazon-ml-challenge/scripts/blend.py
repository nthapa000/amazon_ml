"""Blend OOF predictions, calibrate on OOF, post-process, write the submission.

python blend.py train=train.csv test=test.csv target=price id=sample_id metric=smape \
    tags=lgb1,xgb1,nn1 space=log mult=1 clip=1 snap=0 out=test_out.csv col=price

tags   names used in oof_<tag>.npy / tst_<tag>.npy (from gbdt_cv.py or your NN code)
space  log (geometric blend, for positive skewed targets) | lin | rank (classification scores)
mult   regression: tune one global multiplier on OOF (SMAPE / MAPE usually gain)
thr    binary classification: tune decision threshold on OOF for the metric (F1, acc)
clip   clip predictions to the train target range
snap   round to nearest training target value; kept only if OOF improves
Every step prints OOF before/after; a step that hurts OOF is skipped automatically.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from kv import args, cols, read
from metrics import M, tune_mult, tune_thr, snap, better

a = args({'train': 'train.csv', 'test': 'test.csv', 'target': 'target', 'id': 'sample_id',
          'metric': 'smape', 'tags': '', 'space': 'log', 'mult': True, 'thr': True, 'clip': True,
          'snap': False, 'out': 'test_out.csv', 'col': '', 'task': 'auto', 'pos': ''})
tr, te = read(a['train']), read(a['test'])
fn, mode, kind = M[a['metric']]
task = a['task'] if a['task'] != 'auto' else kind
tags = cols(a['tags'])
if np.load(f'oof_{tags[0]}.npy').ndim == 2:          # multiclass: equal-weight prob average + argmax
    cl, y = np.unique(tr[a['target']].astype(str), return_inverse=True)
    P = sum(np.load(f'oof_{t}.npy') for t in tags) / len(tags)
    Q = sum(np.load(f'tst_{t}.npy') for t in tags) / len(tags)
    print(f'multiclass blend OOF {a["metric"]} = {fn(y, P.argmax(1)):.4f}')
    pd.DataFrame({a['id']: te[a['id']].values, a['col'] or a['target']: cl[Q.argmax(1)]}).to_csv(a['out'], index=False)
    raise SystemExit('wrote ' + a['out'])
O = np.column_stack([np.load(f'oof_{t}.npy') for t in tags])
S = np.column_stack([np.load(f'tst_{t}.npy') for t in tags])
if task == 'reg':
    y = tr[a['target']].values.astype(float)
else:
    cl, y = np.unique(tr[a['target']].astype(str), return_inverse=True)


def score(p):
    if task == 'reg' or a['metric'] == 'auc':
        return fn(y, p)
    return fn(y, (p > .5).astype(int))


def tf(Z):
    if a['space'] == 'log':
        return np.log(np.clip(Z, 1e-9, None))
    if a['space'] == 'rank':
        return np.column_stack([rankdata(z) / len(z) for z in Z.T])
    return Z


def itf(z):
    return np.exp(z) if a['space'] == 'log' else z


for j, t in enumerate(tags):
    print(f'{t:>12}: OOF {a["metric"]} = {score(O[:, j]):.4f}')
if len(tags) > 1:
    print('OOF error correlation (lower = more useful to blend):')
    E = tf(O) - (np.log(y)[:, None] if (task == 'reg' and a['space'] == 'log') else 0)
    print(pd.DataFrame(np.corrcoef(E.T), index=tags, columns=tags).round(3).to_string())

m = len(tags)
if m == 1:
    w = np.ones(1)
else:
    Ot = tf(O)
    sign = 1 if mode == 'min' else -1
    f = lambda w: sign * score(itf(Ot @ (np.abs(w) / np.abs(w).sum())))
    r = minimize(f, np.ones(m) / m, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-4})
    w = np.abs(r.x) / np.abs(r.x).sum()
print('weights:', {t: round(float(x), 3) for t, x in zip(tags, w)})
if a['space'] == 'rank':
    po = tf(O) @ w
    Sr = np.column_stack([rankdata(s) / len(s) for s in S.T]) @ w
    ps = Sr
else:
    po, ps = itf(tf(O) @ w), itf(tf(S) @ w)
best = score(po)
print(f'blend OOF = {best:.4f}')

if task == 'reg':
    if a['mult']:
        k, s = tune_mult(y, po, fn, mode)
        if better(s, best, mode):
            po, ps, best = po * k, ps * k, s
            print(f'multiplier {k:.3f} -> OOF {best:.4f}')
        else:
            print('multiplier: no gain, skipped')
    if a['clip']:
        lo, hi = y.min(), y.max()
        s = score(np.clip(po, lo, hi))
        if not better(best, s, mode):
            po, ps, best = np.clip(po, lo, hi), np.clip(ps, lo, hi), s
            print(f'clip to [{lo:.4g}, {hi:.4g}] -> OOF {best:.4f}')
    if a['snap']:
        s = score(snap(po, y))
        if better(s, best, mode):
            po, ps, best = snap(po, y), snap(ps, y), s
            print(f'snap to train grid -> OOF {best:.4f}')
        else:
            print('snap: no gain, skipped')
    out = ps
elif a['metric'] != 'auc' and O.ndim == 2 and len(np.unique(y)) == 2 and a['thr']:
    t, s = tune_thr(y, po, fn, mode)
    print(f'threshold {t:.2f} -> OOF {s:.4f} (0.5 gave {best:.4f})')
    out = cl[(ps > t).astype(int)]
    best = s
else:
    out = ps
print(f'FINAL OOF {a["metric"]} = {best:.4f}')
col = a['col'] or a['target']
sub = pd.DataFrame({a['id']: te[a['id']].values, col: out})
sub.to_csv(a['out'], index=False)
print('wrote', a['out'], sub.shape, '-> now run sanity.py')
