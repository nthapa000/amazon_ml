"""Final candidate set = learned filter on the first-pass candidates (keys + expansion + ANN): the pairs whose stage-1
score p1 >= thr. The final model's scores are restricted to this set, so matching and the decision rule run only on it.
A pair's final score depends only on its own features, so this equals running the final model on the set alone.

python cands.py s1=b1ka fin=fin5 thr=0.080 out=final
-> W/oof_{out}.parquet (validation pairs of the set, for tuning the decision rule), W/pte_{out}.parquet (test pairs of the
   set = candidate_pairs.tsv, written by decide.py apply)
"""
import os
import sys

import polars as pl

W = os.environ.get('ER_WORK', 'work')
kv = dict(a.split('=', 1) for a in sys.argv[1:])
thr = float(kv['thr'])
for kind, sp in (('oof', 'tsim'), ('pte', 'test')):
    c = pl.scan_parquet(f'{W}/{kind}_{kv["s1"]}.parquet').filter(pl.col('p') >= thr).select('id1', 'id2').collect()
    f = pl.read_parquet(f'{W}/{kind}_{kv["fin"]}.parquet')
    r = f.join(c, on=['id1', 'id2'], how='semi')
    assert r.height == c.height, 'every candidate must carry a final-model score'
    r.write_parquet(f'{W}/{kind}_{kv["out"]}.parquet')
    print(sp, f'p1 >= {thr}: candidates {c.height}, final-model pairs kept {r.height} of {f.height}', flush=True)
