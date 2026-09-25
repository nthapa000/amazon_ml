"""Assemble the expanded candidate set kx = K100 candidates + sibling-expansion pairs.

python kx.py prep split=tsim   -> expc_{split}.parquet (exp pairs, base schema), cand_{split}_kx.parquet (id1,id2), sib_{split}.parquet
python kx.py link split=tsim   -> f_cand_{split}_kx/ = K100 feature parts + expansion feature parts
"""
import glob
import os
import sys

import polars as pl

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')
cmd = sys.argv[1]
kv = dict(a.split('=', 1) for a in sys.argv[2:])
sp = kv['split']
if cmd == 'prep':
    base = pl.read_parquet(f'{W}/cand_{sp}_k100.parquet')
    ex = pl.read_parquet(f'{W}/exp_{sp}.parquet')
    ex.select(base.columns).write_parquet(f'{W}/expc_{sp}.parquet')
    pl.concat([base.select('id1', 'id2'), ex.select('id1', 'id2')]).write_parquet(f'{W}/cand_{sp}_kx.parquet')
    sib = pl.concat([base.select('id1', 'id2', is_exp=pl.lit(0, pl.Int8), e_sib=pl.lit(0, pl.Int8)),
                     ex.select('id1', 'id2', is_exp=pl.lit(1, pl.Int8), e_sib=pl.col('e_sib').cast(pl.Int8))])
    sib.write_parquet(f'{W}/sib_{sp}.parquet')
    print(sp, 'base', base.height, 'exp', ex.height)
else:
    d = f'{W}/f_cand_{sp}_kx'
    os.makedirs(d, exist_ok=True)
    for p in sorted(glob.glob(f'{W}/f_cand_{sp}_k100/*.parquet')):
        os.symlink(os.path.realpath(p), f'{d}/{os.path.basename(p)}')
    for p in sorted(glob.glob(f'{W}/f_expc_{sp}/p_*.parquet')):
        os.symlink(p, f'{d}/c_{os.path.basename(p)}')
    print('linked', len(os.listdir(d)))
