"""Stage-2 context / competition features from stage-1 scores (OOF on train, fold-avg on test).

python ctx.py tag=s1   -> W/ctx_{tag}_train.parquet, W/ctx_{tag}_test.parquet
"""
import os
import sys

import polars as pl

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')


def ctx(d):
    p = pl.col('p')
    rk = p.rank('ordinal', descending=True)
    d = d.with_columns(
        c_p=p,
        c_r1=rk.over('id1').cast(pl.Int16), c_m1=p.max().over('id1'), c_s1=p.sum().over('id1'),
        c_n1h=(p > 0.5).sum().over('id1').cast(pl.Int16), c_n1=pl.len().over('id1').cast(pl.Int16),
        c_r2=rk.over('id2').cast(pl.Int16), c_m2=p.max().over('id2'), c_s2=p.sum().over('id2'),
        c_n2=pl.len().over('id2').cast(pl.Int16), t2=p.top_k(2).min().over('id2'),
        t1b=p.top_k(2).min().over('id1'))
    d = d.with_columns(
        c_oth2=pl.when(pl.col('c_r2') == 1).then(pl.when(pl.col('c_n2') > 1).then(pl.col('t2')).otherwise(0.0))
        .otherwise(pl.col('c_m2')),
        c_gap1=p - pl.col('c_m1'),
        c_2nd1=pl.when(pl.col('c_n1') > 1).then(pl.col('t1b')).otherwise(0.0))
    d = d.with_columns(c_marg2=p - pl.col('c_oth2'), c_rest1=pl.col('c_s1') - p)
    return d.drop('p', 't2', 't1b', *[c for c in d.columns if c in ('y',)])


if __name__ == '__main__':
    kv = dict(a.split('=', 1) for a in sys.argv[1:])
    tag = kv['tag']
    ctx(pl.read_parquet(f'{W}/oof_{tag}.parquet').select('id1', 'id2', 'p')).write_parquet(f'{W}/ctx_{tag}_train.parquet')
    if os.path.exists(f'{W}/pte_{tag}.parquet'):
        ctx(pl.read_parquet(f'{W}/pte_{tag}.parquet')).write_parquet(f'{W}/ctx_{tag}_test.parquet')
    print('done')
