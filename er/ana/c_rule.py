"""Feasibility of France-only name-conflict rules (C'): pattern prevalence among predicted links per country (test),
precision of those links on labelled US/India validation, and the F0.5 effect of dropping them on validation.
python ana/c_rule.py tag=fin2bc      (needs W/f3_tsim, W/f3_test from src/feats3.py)
"""
import os
import sys
import json

sys.path.insert(0, '/csehome/m25csa019/amazon_ml/er/src')
import polars as pl  # noqa: E402
from block import enc, gt_path  # noqa: E402
from decide import select, score  # noqa: E402

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')
kv = dict(a.split('=', 1) for a in sys.argv[1:])
tag = kv.get('tag', 'fin2bc')
cfg = json.load(open(f'{W}/dec_{tag}.json'))
RULES = {
    'swap_common': (pl.col('swap') == 1) & (pl.col('fq1') > -3.5) & (pl.col('fq2') > -3.5),  # real-word swap
    'swap_any': pl.col('swap') == 1,
    'nm_eq_stconf': pl.col('nm_eq_stc') == 1,                                                 # same name, other street
    'sib_swap': pl.col('sib_swap') == 1,                                                      # conflicts with siblings
    'swap_and_sib': (pl.col('swap') == 1) & (pl.col('sib_swap') == 1),
}


def pats(d):
    return d.with_columns([r.fill_null(False).alias(n) for n, r in RULES.items()])


# ---- validation (US/India, labelled)
s1 = pl.read_parquet(f'{W}/n_tsim_s1.parquet', columns=['id', 'country']).select(enc('id').alias('s1'), 'country')
gt = pl.read_parquet(gt_path('tsim')).select(enc('s1').alias('s1'), enc('id2').alias('id2'))
o = pl.read_parquet(f'{W}/oof_{tag}.parquet').select('id1', 'id2', 'p')
sel = select(o, cfg).join(o, on=['id1', 'id2'])
v = pats(sel.join(pl.read_parquet(f'{W}/f3_tsim.parquet'), on=['id1', 'id2'], how='left'))
v = v.join(gt.rename({'s1': 'id1'}).with_columns(y=pl.lit(1, pl.Int8)), on=['id1', 'id2'], how='left').with_columns(
    pl.col('y').fill_null(0))
base = score(sel.select('id1', 'id2'), s1, gt)['ALL']
print(f'VALIDATION (US/India) selected links {v.height}, precision {v["y"].mean():.4f}, F0.5 {base:.6f}')
for n in RULES:
    g = v.filter(pl.col(n))
    kept = v.filter(~pl.col(n)).select('id1', 'id2')
    f = score(kept, s1, gt)['ALL']
    gb = g.filter(pl.col('p') < 0.9)
    print(f'  {n:13s} links {g.height:7d} ({g.height / v.height:.3%}) precision {g["y"].mean() if g.height else 0:.3f}'
          f' | p<0.9: {gb.height:6d} precision {gb["y"].mean() if gb.height else 0:.3f}'
          f' | F0.5 if dropped {f:.6f} ({f - base:+.6f})', flush=True)

# ---- test: prevalence per country among predicted links
ct = pl.read_parquet(f'{W}/n_test_s1.parquet', columns=['id', 'country']).select(enc('id').alias('id1'), 'country')
pt = pl.read_parquet(f'{W}/pte_{tag}.parquet').select('id1', 'id2', 'p')
st = select(pt, cfg).join(pt, on=['id1', 'id2']).join(ct, on='id1')
t = pats(st.join(pl.read_parquet(f'{W}/f3_test.parquet'), on=['id1', 'id2'], how='left'))
vv = v.join(s1.rename({'s1': 'id1'}), on='id1')
print('\nPREVALENCE among predicted links (share of links; share of links with p<0.9 in brackets)')
rows = []
for name, d in [('val ' + c, vv.filter(pl.col('country') == c)) for c in ('US', 'India')] + \
               [('test ' + c, t.filter(pl.col('country') == c)) for c in ('US', 'India', 'France')]:
    b = d.filter(pl.col('p') < 0.9)
    rows.append(f'  {name:12s} links {d.height:8d} ' + ' '.join(
        f'{n}={d[n].mean():.3%}({b[n].mean() if b.height else 0:.2%})' for n in RULES))
print('\n'.join(rows))
fr = t.filter(pl.col('country') == 'France')
nfr = ct.filter(pl.col('country') == 'France').height
print(f'\nFRANCE: {fr.height} predicted links over {nfr} S1')
for n in RULES:
    g = fr.filter(pl.col(n))
    print(f'  {n:13s} would drop {g.height:6d} links ({g.height / fr.height:.2%}), touching {g["id1"].n_unique():6d} S1'
          f' ({g["id1"].n_unique() / nfr:.2%}); p<0.9 among them {g.filter(pl.col("p") < 0.9).height}', flush=True)
