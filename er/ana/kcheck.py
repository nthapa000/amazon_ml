"""Gate for an imported stage-2 score file before blending: exact row counts, 100% (id1,id2) key overlap with a reference
tag, identical tsim labels, p in [0,1]; prints correlations with other tags. Exits 1 on any failure.
python ana/kcheck.py tag=kfx ref=fin6s others=kcxQ
"""
import os
import sys

import polars as pl

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')
kv = dict(a.split('=', 1) for a in sys.argv[1:])
t, ref = kv['tag'], kv.get('ref', 'fin6s')
ok = True
for k, n in (('oof', 8194747), ('pte', 9187520)):
    a = pl.read_parquet(f'{W}/{k}_{ref}.parquet', columns=['id1', 'id2'])
    b = pl.read_parquet(f'{W}/{k}_{t}.parquet')
    ov = a.join(b.select('id1', 'id2'), on=['id1', 'id2']).height
    good = b.height == n and ov == n and b['p'].is_between(0, 1).all() and b['p'].null_count() == 0
    ok &= good
    print(k, t, 'rows', b.height, 'overlap', f'{ov / n:.6f}', 'p', round(float(b['p'].min()), 6), round(float(b['p'].max()), 6),
          'OK' if good else 'FAIL', flush=True)
o = pl.read_parquet(f'{W}/oof_{t}.parquet')
if 'y' in o.columns:
    x = o.join(pl.read_parquet(f'{W}/oof_{ref}.parquet', columns=['id1', 'id2', 'y']), on=['id1', 'id2'], suffix='_r')
    same = bool((x['y'] == x['y_r']).all())
    ok &= same
    print('tsim labels identical to', ref, same, flush=True)
for r in [ref] + [x for x in kv.get('others', '').split('+') if x]:
    c = o.select('id1', 'id2', 'p').join(pl.read_parquet(f'{W}/oof_{r}.parquet', columns=['id1', 'id2', 'p']), on=['id1', 'id2'],
                                         suffix='_r')
    print(f'corr oof p {t} vs {r}: {c.select(pl.corr("p", "p_r")).item():.4f}', flush=True)
sys.exit(0 if ok else 1)
