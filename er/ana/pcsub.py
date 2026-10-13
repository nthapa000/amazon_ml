"""Per-country cascade keep-lists for plan item L3 (country-specific stage 2): the stage-1 cascade pairs (p1 >= thr) whose
S1 is in one country, so `src/train.py sub=oof_b1ka_<c>:pte_b1ka_<c>:<thr>` trains / scores that country only
(same folds: fold = hash(id1)).

python ana/pcsub.py ctry=US [s1=b1ka] [thr=0.001]  -> W/oof_<s1>_<c>.parquet, W/pte_<s1>_<c>.parquet  (id1, id2, p)
"""
import os
import sys

import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from block import enc  # noqa: E402

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')
kv = dict(a.split('=', 1) for a in sys.argv[1:])
c, s, thr = kv['ctry'], kv.get('s1', 'b1ka'), float(kv.get('thr', 0.001))
for kind, sp in (('oof', 'tsim'), ('pte', 'test')):
    ids = pl.read_parquet(f'{W}/n_{sp}_s1.parquet', columns=['id', 'country']).filter(pl.col('country') == c)
    ids = ids.select(enc('id').alias('id1'))
    d = (pl.scan_parquet(f'{W}/{kind}_{s}.parquet').filter(pl.col('p') >= thr).select('id1', 'id2', 'p')
         .join(ids.lazy(), on='id1', how='semi').collect())
    d.write_parquet(f'{W}/{kind}_{s}_{c}.parquet')
    print(kind, sp, c, d.height, flush=True)
