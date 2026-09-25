"""Blend several scored versions (seeds / models / candidate sets) in logit space, weights tuned on OOF.

python blend.py tags=w2a,w2b,w2c out=w2s split=tsim [grid=1]
- pairs missing from a member (different candidate sets) use the members that have them (weights renormalized)
- grid=1: search the weight of the first member vs the mean of the rest on OOF macro F0.5
-> W/oof_{out}.parquet, W/pte_{out}.parquet   (then run decide.py tune/apply on tag=out)
"""
import os
import sys
import json

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
from block import enc, gt_path  # noqa: E402
from decide import select, score  # noqa: E402

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def combine(kind, tags, w):
    d = None
    for i, t in enumerate(tags):
        x = pl.read_parquet(f'{W}/{kind}_{t}.parquet', columns=['id1', 'id2', 'p']).rename({'p': f'p{i}'})
        d = x if d is None else d.join(x, on=['id1', 'id2'], how='full', coalesce=True)
    L = np.stack([logit(d[f'p{i}'].fill_null(0.5).to_numpy()) for i in range(len(tags))], 1)
    M = np.stack([d[f'p{i}'].is_not_null().to_numpy() for i in range(len(tags))], 1).astype(np.float64)
    ww = M * np.asarray(w)[None, :]
    z = (L * ww).sum(1) / np.maximum(ww.sum(1), 1e-9)
    return d.select('id1', 'id2').with_columns(p=pl.Series(1 / (1 + np.exp(-z)), dtype=pl.Float32))


def main(kv):
    tags, out, split = kv['tags'].replace('+', ',').split(','), kv['out'], kv.get('split', 'tsim')   # '+' avoids sbatch comma splitting
    cfg = dict(method='ef', margin=float(kv.get('margin', 0.2)), m=0.0, a=1.0)
    s1 = pl.read_parquet(f'{W}/n_{split}_s1.parquet', columns=['id', 'country']).select(enc('id').alias('s1'), 'country')
    gt = pl.read_parquet(gt_path(split)).select(enc('s1').alias('s1'), enc('id2').alias('id2'))
    for t in tags:                                   # members alone, same decision rule
        o = pl.read_parquet(f'{W}/oof_{t}.parquet', columns=['id1', 'id2', 'p'])
        print(f'member {t}: {score(select(o, cfg), s1, gt, False)["ALL"]:.5f}', flush=True)
    n = len(tags)
    cands = [np.ones(n) / n]
    if kv.get('grid') == '1' and n > 1:
        cands = [np.array([a] + [(1 - a) / (n - 1)] * (n - 1)) for a in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)]
    best = (-1, None)
    for w in cands:
        o = combine('oof', tags, w)
        f = score(select(o, cfg), s1, gt, False)['ALL']
        print(f'weights {np.round(w, 3).tolist()}: {f:.5f}', flush=True)
        if f > best[0]:
            best = (f, w)
    w = best[1]
    print('BEST weights', np.round(w, 3).tolist(), f'{best[0]:.5f}', flush=True)
    o = combine('oof', tags, w)
    y = pl.read_parquet(gt_path(split)).select(enc('s1').alias('id1'), enc('id2').alias('id2'), y=pl.lit(1, pl.Int8))
    o.join(y, on=['id1', 'id2'], how='left').with_columns(pl.col('y').fill_null(0)).write_parquet(f'{W}/oof_{out}.parquet')
    combine('pte', tags, w).write_parquet(f'{W}/pte_{out}.parquet')
    json.dump(dict(tags=tags, weights=np.asarray(w).tolist()), open(f'{W}/blend_{out}.json', 'w'))


if __name__ == '__main__':
    main(dict(a.split('=', 1) for a in sys.argv[1:]))
