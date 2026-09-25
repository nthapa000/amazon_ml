"""Calibration check (playbook S5): reliability of OOF probabilities, overall and per country.
python ana/calib.py tag=w2
"""
import sys

import polars as pl

sys.path.insert(0, '/csehome/m25csa019/amazon_ml/er/src')
from block import enc  # noqa: E402

W = '/scratch/m25csa019/er_work'
kv = dict(a.split('=', 1) for a in sys.argv[1:])
o = pl.read_parquet(f'{W}/oof_{kv["tag"]}.parquet')
c = pl.read_parquet(f'{W}/n_tsim_s1.parquet', columns=['id', 'country']).select(enc('id').alias('id1'), 'country')
o = o.join(c, on='id1')
edges = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0001]
o = o.with_columns(bin=pl.col('p').cut(edges[1:-1], left_closed=True))
with pl.Config(tbl_rows=40):
    print(o.group_by('bin').agg(pl.len(), pl.col('p').mean().alias('mean_p'), pl.col('y').mean().alias('rate'))
          .sort('mean_p'))
    print(o.group_by('country', 'bin').agg(pl.len(), pl.col('p').mean().alias('mean_p'), pl.col('y').mean().alias('rate'))
          .filter(pl.col('mean_p') > 0.2).sort('country', 'mean_p'))
