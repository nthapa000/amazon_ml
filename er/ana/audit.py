"""Overfitting / drift audit (playbook 4H).
1) adversarial validation: can a classifier tell test candidate pairs from tsim (train) pairs? per country.
2) transliteration-dictionary coverage: native-script tokens covered on train vs test (in-sample learning risk).
python ana/audit.py
"""
import glob
import json
import sys

import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/csehome/m25csa019/amazon_ml/er/src')
from block import enc  # noqa: E402
import norm  # noqa: E402

W = '/scratch/m25csa019/er_work'
DROP = {'id1', 'id2', 'kbits', 'y', 'side', 'country'}


def sample(split, n_parts=6, frac=0.25):
    fs = sorted(glob.glob(f'{W}/f_cand_{split}_kx/a_*.parquet'))[:n_parts]
    d = pl.concat([pl.read_parquet(f) for f in fs], how='vertical_relaxed').sample(fraction=frac, seed=1)
    c = pl.read_parquet(f'{W}/n_{split}_s1.parquet', columns=['id', 'country']).select(enc('id').alias('id1'), 'country')
    return d.join(c, on='id1')


tr, te = sample('tsim'), sample('test')
feats = [c for c in tr.columns if c not in DROP]


def adv(a, b, name):
    d = pl.concat([a.with_columns(side=pl.lit(0)), b.with_columns(side=pl.lit(1))], how='vertical_relaxed')
    d = d.sample(fraction=1.0, seed=2, shuffle=True)
    X, y = d.select(feats).to_numpy().astype(np.float32), d['side'].to_numpy()
    h = len(y) // 2
    m = lgb.train(dict(objective='binary', learning_rate=0.1, num_leaves=63, verbose=-1, num_threads=16),
                  lgb.Dataset(X[:h], y[:h]), 200)
    auc = roc_auc_score(y[h:], m.predict(X[h:]))
    imp = sorted(zip(feats, m.feature_importance('gain')), key=lambda x: -x[1])[:8]
    tot = sum(g for _, g in zip(feats, m.feature_importance('gain')))
    print(f'[{name}] adversarial AUC {auc:.3f}  top drift: ' + ', '.join(f'{f}={g / tot:.2f}' for f, g in imp), flush=True)


adv(tr.filter(pl.col('country') == 'US'), te.filter(pl.col('country') == 'US'), 'US tsim vs US test')
adv(tr.filter(pl.col('country') == 'India'), te.filter(pl.col('country') == 'India'), 'India tsim vs India test')
adv(tr, te.filter(pl.col('country') == 'France'), 'US+India tsim vs France test')
adv(tr.filter(pl.col('country') == 'US'), tr.filter(pl.col('country') == 'India'), 'reference: US vs India (tsim)')

# 2) dictionary coverage
td = json.load(open(f'{W}/tdict.json'))
for sp in ('train', 'test'):
    names = pl.concat([pl.read_parquet(f'{W}/raw_{sp}_s{k}.parquet', columns=['name', 'country']) for k in (2, 3)])
    names = names.filter(pl.col('country') == 'India').sample(n=300000, seed=3)['name'].to_list()
    tot = cov = 0
    for n in names:
        for t in norm.clean_name_toks(norm.split_name(n)[0]):
            if norm.has_ind(t):
                tot += 1
                cov += t in td
    print(f'dict coverage {sp}: {cov / max(tot, 1):.4f} of {tot} native tokens', flush=True)
