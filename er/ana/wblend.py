"""Team blend of several stage-2 score files (logit space) with a weight search over the simplex, tuned on tsim OOF.
Countries unseen in training (no labels) can get their own fixed weights (e.g. equal), because weights tuned on US/India
say nothing about them. No src/ edits.

python ana/wblend.py tags=kcxQ+fin6s+kX out=tb [step=0.25 fine=0.05] [unseen=equal | unseen=0.5:0.3:0.2] [margin=0.3]
-> W/oof_<out>.parquet (id1,id2,y,p), W/pte_<out>.parquet (id1,id2,p), W/blend_<out>.json
"""
import os
import sys
import json
import itertools

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from block import enc, gt_path  # noqa: E402
from decide import select, score, unseen  # noqa: E402

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')


def load(kind, tags):
    d = None
    for i, t in enumerate(tags):
        x = pl.read_parquet(f'{W}/{kind}_{t}.parquet', columns=['id1', 'id2', 'p']).rename({'p': f'p{i}'})
        d = x if d is None else d.join(x, on=['id1', 'id2'], how='full', coalesce=True)
    P = np.stack([d[f'p{i}'].fill_null(0.5).to_numpy() for i in range(len(tags))], 1)
    L = np.log(np.clip(P, 1e-6, 1 - 1e-6) / (1 - np.clip(P, 1e-6, 1 - 1e-6)))
    M = np.stack([d[f'p{i}'].is_not_null().to_numpy() for i in range(len(tags))], 1).astype(np.float64)
    return d.select('id1', 'id2'), L, M


def mix(L, M, w):
    ww = M * np.asarray(w)[None, :]
    z = (L * ww).sum(1) / np.maximum(ww.sum(1), 1e-9)
    return 1 / (1 + np.exp(-z))


def simplex(n, step):
    k = int(round(1 / step))
    return [np.array(c) / k for c in itertools.product(range(k + 1), repeat=n) if sum(c) == k]


def main(kv):
    tags, out = kv['tags'].replace(',', '+').split('+'), kv['out']
    cfg = dict(method='ef', margin=float(kv.get('margin', 0.3)), m=0.0, a=1.0)
    s1 = pl.read_parquet(f'{W}/n_tsim_s1.parquet', columns=['id', 'country']).select(enc('id').alias('s1'), 'country')
    gt = pl.read_parquet(gt_path('tsim')).select(enc('s1').alias('s1'), enc('id2').alias('id2'))
    ids, L, M = load('oof', tags)
    cache = {}

    def f(w):
        key = tuple(np.round(w, 4))
        if key not in cache:
            o = ids.with_columns(p=pl.Series(mix(L, M, w), dtype=pl.Float32))
            cache[key] = score(select(o, cfg), s1, gt, False)['ALL']
            print(f'  w {np.round(w, 3).tolist()}: {cache[key]:.6f}', flush=True)
        return cache[key]
    n = len(tags)
    for i in range(n):
        f(np.eye(n)[i])
    best = max(simplex(n, float(kv.get('step', 0.25))), key=f)
    fine = float(kv.get('fine', 0.05))
    for _ in range(2):                                   # local refinement around the best point
        cand = [best + fine * (np.eye(n)[i] - np.eye(n)[j]) for i in range(n) for j in range(n) if i != j]
        cand = [c for c in cand if c.min() >= -1e-9]
        b2 = max(cand + [best], key=f)
        if f(b2) <= f(best):
            break
        best = b2
    print(f'BEST seen-country weights {dict(zip(tags, np.round(best, 3).tolist()))}: {f(best):.6f}', flush=True)
    un = kv.get('unseen')
    wu = best if un is None else (np.ones(n) / n if un == 'equal' else np.array([float(x) for x in un.split(':')]))
    wu = wu / wu.sum()
    print(f'unseen-country weights {dict(zip(tags, np.round(wu, 3).tolist()))}', flush=True)
    y = pl.read_parquet(gt_path('tsim')).select(enc('s1').alias('id1'), enc('id2').alias('id2'), y=pl.lit(1, pl.Int8))
    (ids.with_columns(p=pl.Series(mix(L, M, best), dtype=pl.Float32)).join(y, on=['id1', 'id2'], how='left')
     .with_columns(pl.col('y').fill_null(0)).write_parquet(f'{W}/oof_{out}.parquet'))
    tids, TL, TM = load('pte', tags)
    t1 = pl.read_parquet(f'{W}/n_test_s1.parquet', columns=['id', 'country'])
    cs = unseen(t1)
    un_ids = t1.filter(pl.col('country').is_in(cs)).select(enc('id').alias('id1'))
    is_un = tids.join(un_ids.with_columns(u=pl.lit(True)), on='id1', how='left')['u'].fill_null(False).to_numpy()
    p = np.where(is_un, mix(TL, TM, wu), mix(TL, TM, best))
    tids.with_columns(p=pl.Series(p, dtype=pl.Float32)).write_parquet(f'{W}/pte_{out}.parquet')
    json.dump(dict(tags=tags, weights=best.tolist(), unseen_weights=wu.tolist(), unseen_countries=cs, oof=f(best)),
              open(f'{W}/blend_{out}.json', 'w'))
    print(f'test rows {tids.height}, unseen-country rows {int(is_un.sum())}', flush=True)


if __name__ == '__main__':
    main(dict(a.split('=', 1) for a in sys.argv[1:]))
