"""Compare two final models on the validation folds not used for tuning (fold != skip), each with its own decision config.
Decisions are made on the full OOF (one-to-one competes across folds), then scored per fold subset.
python ana/cmp_folds.py a=fin b=fin2b skip=0"""
import os
import sys
import json

sys.path.insert(0, '/csehome/m25csa019/amazon_ml/er/src')
import polars as pl  # noqa: E402
from block import enc, gt_path  # noqa: E402
from decide import select, score  # noqa: E402

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')
kv = dict(a.split('=', 1) for a in sys.argv[1:])
skip = int(kv.get('skip', 0))
s1 = (pl.read_parquet(f'{W}/n_tsim_s1.parquet', columns=['id', 'country']).select(enc('id').alias('s1'), 'country')
      .with_columns(fold=(pl.col('s1').hash(seed=7) % 5).cast(pl.Int8)))
gt = pl.read_parquet(gt_path('tsim')).select(enc('s1').alias('s1'), enc('id2').alias('id2'))
res = {}
for t in (kv['a'], kv['b']):
    cfg = json.load(open(f'{W}/dec_{t}.json'))
    sel = select(pl.read_parquet(f'{W}/oof_{t}.parquet').select('id1', 'id2', 'p'), cfg)
    for name, sub in (('all folds', s1), (f'folds != {skip}', s1.filter(pl.col('fold') != skip)),
                      (f'fold {skip} only', s1.filter(pl.col('fold') == skip))):
        ids = sub.select(pl.col('s1').alias('id1'))
        r = score(sel.join(ids, on='id1', how='semi'), sub.drop('fold'), gt.join(sub.select('s1'), on='s1', how='semi'))
        res[(t, name)] = r
    per_fold = []
    for f in range(5):
        sub = s1.filter(pl.col('fold') == f)
        r = score(sel.join(sub.select(pl.col('s1').alias('id1')), on='id1', how='semi'), sub.drop('fold'),
                  gt.join(sub.select('s1'), on='s1', how='semi'), by_country=False)
        per_fold.append(r['ALL'])
    res[(t, 'per fold')] = per_fold
a, b = kv['a'], kv['b']
for name in ('all folds', f'folds != {skip}', f'fold {skip} only'):
    ra, rb = res[(a, name)], res[(b, name)]
    print(f'{name:14s} {a} {ra["ALL"]:.6f}  {b} {rb["ALL"]:.6f}  diff {rb["ALL"] - ra["ALL"]:+.6f}'
          f'  (US {rb["US"] - ra["US"]:+.6f}, India {rb["India"] - ra["India"]:+.6f})', flush=True)
print('per fold diff', ' '.join(f'{y - x:+.6f}' for x, y in zip(res[(a, 'per fold')], res[(b, 'per fold')])), flush=True)
