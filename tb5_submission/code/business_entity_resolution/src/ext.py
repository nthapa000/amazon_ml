"""Top-K caps on the uncapped blocking output (block.py gen): keep, for every S1, its k1 best-scoring key candidates
and, for every S2/S3 record, its k2 best S1 (block.cap), over all countries of the split.

python ext.py make split=tsim k1=100 k2=10   -> W/cand_{split}_k{k1}.parquet
"""
import glob
import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
from block import cap  # noqa: E402

W = os.environ.get('ER_WORK', 'work')
kv = dict(a.split('=', 1) for a in sys.argv[2:])
split, k1, k2 = kv['split'], int(kv.get('k1', 100)), int(kv.get('k2', 10))
name = f'cand_{split}_k{k1}'

if sys.argv[1] == 'make':
    c0 = pl.concat([pl.read_parquet(p) for p in sorted(glob.glob(f'{W}/cand0_{split}_*.parquet'))])
    full = cap(c0, k1, k2)
    del c0
    full.write_parquet(f'{W}/{name}.parquet')
    print(split, 'candidates', full.height, 'per S1', round(full.height / max(full['id1'].n_unique(), 1), 1))
