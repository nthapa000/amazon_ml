"""Extra pair features computed with polars (cheap, whole-split statistics; no labels used).

- n_c1   : #S1 records (same country) with the identical core name as the S1 side  (generic-name risk)
- n_c2   : #S2/S3 records with the identical core name as the S2/S3 side
- n_ca1  : #S1 records sharing the identical normalized address
- v2_min / v2_mean : log(1+df) in S1 core-name vocabulary of the S2/S3 core tokens (0 => invented / DBA name)
- same_core_cands  : #candidates of this S1 whose core equals the S2/S3 core (duplicates in the list)

python xf.py split=train cand=cand_train  -> W/xf_{split}.parquet
"""
import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
from block import enc  # noqa: E402

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')


def main(split, cand, out=None):
    c = ['id', 'country', 'core', 'ad']
    s1 = pl.read_parquet(f'{W}/n_{split}_s1.parquet', columns=c).with_columns(id=enc('id'))
    s23 = pl.concat([pl.read_parquet(f'{W}/n_{split}_s{k}.parquet', columns=c) for k in (2, 3)]).with_columns(id=enc('id'))
    s1 = s1.with_columns(n_c1=pl.len().over('country', 'core').cast(pl.Int32),
                         n_ca1=pl.len().over('country', 'ad').cast(pl.Int32))
    s23 = s23.with_columns(n_c2=pl.len().over('country', 'core').cast(pl.Int32))
    voc = (s1.select('country', pl.col('core').str.split(' ').alias('t')).explode('t')
           .group_by('country', 't').len('df'))
    tk = (s23.select('id', 'country', pl.col('core').str.split(' ').alias('t')).explode('t')
          .join(voc, on=['country', 't'], how='left').with_columns((pl.col('df').fill_null(0) + 1).log().alias('l'))
          .group_by('id').agg(pl.col('l').min().alias('v2_min'), pl.col('l').mean().alias('v2_mean')))
    s23 = s23.join(tk, on='id', how='left')
    pr = pl.read_parquet(f'{W}/{cand}.parquet', columns=['id1', 'id2'])
    d = (pr.join(s1.select(pl.col('id').alias('id1'), pl.col('core').alias('c1'), 'n_c1', 'n_ca1'), on='id1', how='left')
         .join(s23.select(pl.col('id').alias('id2'), pl.col('core').alias('c2'), 'n_c2', 'v2_min', 'v2_mean'),
               on='id2', how='left'))
    d = d.with_columns(same_core_cands=pl.len().over('id1', 'c2').cast(pl.Int16))
    d.drop('c1', 'c2').write_parquet(f'{W}/{out or "xf_" + split}.parquet')
    print(d.describe())


if __name__ == '__main__':
    kv = dict(a.split('=', 1) for a in sys.argv[1:])
    main(kv['split'], kv['cand'], kv.get('out'))
