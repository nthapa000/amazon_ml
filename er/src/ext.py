"""Enlarge the candidate set (higher top-K caps) reusing the stored uncapped blocking output.

python ext.py make split=tsim k1=100 k2=10   -> cand_{split}_k{k1}.parquet (full) + cand_{split}_k{k1}new.parquet (only new pairs)
  (then: feats.py split=.. cand=cand_{split}_k{k1}new)
python ext.py link split=tsim k1=100         -> f_cand_{split}_k{k1}/ = old feature parts + new feature parts (symlinks)
"""
import glob
import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
from block import cap  # noqa: E402

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')
kv = dict(a.split('=', 1) for a in sys.argv[2:])
split, k1, k2 = kv['split'], int(kv.get('k1', 100)), int(kv.get('k2', 10))
name = f'cand_{split}_k{k1}'

if sys.argv[1] == 'make':
    c0 = pl.concat([pl.read_parquet(p) for p in sorted(glob.glob(f'{W}/cand0_{split}_*.parquet'))])
    full = cap(c0, k1, k2)
    del c0
    old = pl.read_parquet(f'{W}/cand_{split}.parquet', columns=['id1', 'id2'])
    new = full.join(old, on=['id1', 'id2'], how='anti')
    full.write_parquet(f'{W}/{name}.parquet')
    new.write_parquet(f'{W}/{name}new.parquet')
    print(split, 'old', old.height, 'full', full.height, 'new', new.height)
else:
    d = f'{W}/f_{name}'
    os.makedirs(d, exist_ok=True)
    for p in sorted(glob.glob(f'{W}/f_cand_{split}/p_*.parquet')):
        os.symlink(p, f'{d}/a_{os.path.basename(p)}')
    for p in sorted(glob.glob(f'{W}/f_{name}new/p_*.parquet')):
        os.symlink(p, f'{d}/b_{os.path.basename(p)}')
    print('linked', len(os.listdir(d)))
