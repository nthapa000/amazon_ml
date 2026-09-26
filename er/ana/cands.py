"""Candidate-set size vs quality for the final candidate file.

The final candidate set = stage-1 filtered pairs. For each cut (stage-1 p1 >= t, or top-K by p1 per S1 among p1 >= 0.001):
candidates per S1 (tsim and test, per country), recall ceiling, perfect-matcher F0.5 on the set, and the final model's
OOF F0.5 when matching only inside the set (decide.select with the tuned config of the final model).
python ana/cands.py s1=b1ka fin=fin      (ER_WORK = work dir)
"""
import os
import sys
import json

sys.path.insert(0, '/csehome/m25csa019/amazon_ml/er/src')
import polars as pl  # noqa: E402
from block import enc, gt_path  # noqa: E402
from decide import select, score  # noqa: E402

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')
kv = dict(a.split('=', 1) for a in sys.argv[1:])
st, ft = kv.get('s1', 'b1ka'), kv.get('fin', 'fin')
o1 = pl.read_parquet(f'{W}/oof_{st}.parquet', columns=['id1', 'id2', 'p']).rename({'p': 'p1'})
t1 = pl.read_parquet(f'{W}/pte_{st}.parquet', columns=['id1', 'id2', 'p']).rename({'p': 'p1'})
of = pl.read_parquet(f'{W}/oof_{ft}.parquet').select('id1', 'id2', 'p')
cfg = json.load(open(f'{W}/dec_{ft}.json'))
s1 = pl.read_parquet(f'{W}/n_tsim_s1.parquet', columns=['id', 'country']).select(enc('id').alias('s1'), 'country')
ct = pl.read_parquet(f'{W}/n_test_s1.parquet', columns=['id', 'country']).select(enc('id').alias('id1'), 'country')
gt = pl.read_parquet(gt_path('tsim')).select(enc('s1').alias('s1'), enc('id2').alias('id2'))
gp = gt.rename({'s1': 'id1'})
ntr, nte, ng = s1.height, ct.height, gt.height
base = score(select(of, cfg), s1, gt)
print(f'tsim S1 {ntr}  test S1 {nte}  true pairs {ng}  base OOF {base["ALL"]:.6f}', flush=True)
print(f'first-pass set: {o1.height / ntr:.1f} cand/S1 tsim, {t1.height / nte:.1f} test, '
      f'recall {o1.join(gp, on=["id1", "id2"], how="semi").height / ng:.5f}', flush=True)
nct = ct.group_by('country').len('n')


def report(name, c, cte):
    rec = c.join(gp, on=['id1', 'id2'], how='semi')
    ceil = score(rec, s1, gt, by_country=False)['ALL']
    r = score(select(of.join(c, on=['id1', 'id2'], how='semi'), cfg), s1, gt)
    pc = (cte.join(ct, on='id1').group_by('country').len('c').join(nct, on='country')
          .with_columns(r=pl.col('c') / pl.col('n')).sort('country'))
    pcs = ' '.join(f'{a}={b:.2f}' for a, b in zip(pc['country'], pc['r']))
    print(f'{name:12s} tsim {c.height / ntr:5.2f}  test {cte.height / nte:5.2f} ({pcs})  recall {rec.height / ng:.5f}'
          f'  ceilF {ceil:.5f}  F {r["ALL"]:.6f} (US {r["US"]:.6f} IN {r["India"]:.6f})  dF {r["ALL"] - base["ALL"]:+.6f}',
          flush=True)


for t in (0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1):
    report(f'p1>={t}', o1.filter(pl.col('p1') >= t).select('id1', 'id2'),
           t1.filter(pl.col('p1') >= t).select('id1', 'id2'))
k1 = o1.filter(pl.col('p1') >= 0.001).with_columns(r=pl.col('p1').rank('ordinal', descending=True).over('id1'))
k2 = t1.filter(pl.col('p1') >= 0.001).with_columns(r=pl.col('p1').rank('ordinal', descending=True).over('id1'))
for k in (3, 4, 5, 6, 8, 10):
    report(f'top{k}', k1.filter(pl.col('r') <= k).select('id1', 'id2'), k2.filter(pl.col('r') <= k).select('id1', 'id2'))
