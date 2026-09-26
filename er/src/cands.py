"""Final candidate set = learned filter on the first-pass candidates (keys + expansion + ANN): pairs whose stage-1 score
p1 >= thr. Writes W/cand_test_final.parquet (id1, id2; the content of candidate_pairs.tsv) and restricts the final model's
predictions to the set, so matching runs only inside it: W/oof_{out}.parquet (tsim, for decision tuning), W/pte_{out}.parquet.
thr 0.005 -> 4.47 candidates per S1 on test, recall ceiling 0.9884, OOF F0.5 -0.00004 vs p1 >= 0.001 (ana/cands.py).

python cands.py s1=b1ka fin=fin thr=0.005 out=finc
"""
import os
import sys

import polars as pl

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')
kv = dict(a.split('=', 1) for a in sys.argv[1:])
thr = float(kv.get('thr', 0.005))
for kind, sp in (('oof', 'tsim'), ('pte', 'test')):
    c = pl.scan_parquet(f'{W}/{kind}_{kv["s1"]}.parquet').filter(pl.col('p') >= thr).select('id1', 'id2').collect()
    f = pl.read_parquet(f'{W}/{kind}_{kv["fin"]}.parquet')
    r = f.join(c, on=['id1', 'id2'], how='semi')
    assert r.height == c.height, 'every candidate must carry a final-model score'
    r.write_parquet(f'{W}/{kind}_{kv["out"]}.parquet')
    if sp == 'test':
        c.write_parquet(f'{W}/cand_test_final.parquet')
    print(sp, f'p1 >= {thr}: candidates {c.height}, final-model pairs kept {r.height} of {f.height}', flush=True)
