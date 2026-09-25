"""Cascade parity: decision on all pairs vs only pairs with stage-1 p >= thr (must be ~identical)."""
import sys, json
sys.path.insert(0, '/csehome/m25csa019/amazon_ml/er/src')
import polars as pl
from block import enc, gt_path
from decide import select, score
W = '/scratch/m25csa019/er_work'
kv = dict(a.split('=', 1) for a in sys.argv[1:])
tag, s1tag, thr = kv.get('tag', 'w2'), kv.get('stage1', 'w1'), float(kv.get('thr', 1e-3))
cfg = json.load(open(f'{W}/dec_{tag}.json'))
o = pl.read_parquet(f'{W}/oof_{tag}.parquet', columns=['id1', 'id2', 'p'])
k = pl.scan_parquet(f'{W}/oof_{s1tag}.parquet').filter(pl.col('p') >= thr).select('id1', 'id2').collect()
s1 = pl.read_parquet(f'{W}/n_tsim_s1.parquet', columns=['id', 'country']).select(enc('id').alias('s1'), 'country')
gt = pl.read_parquet(gt_path('tsim')).select(enc('s1').alias('s1'), enc('id2').alias('id2'))
full = score(select(o, cfg), s1, gt)['ALL']
sub = score(select(o.join(k, on=['id1', 'id2'], how='semi'), cfg), s1, gt)['ALL']
print(f'{tag}: all pairs {full:.6f}   cascade subset (stage-1 {s1tag} p>={thr}) {sub:.6f}   diff {sub - full:+.6f}')
