"""Content-identical S2/S3 twins (same raw name + address + country): does the decision split them?
Organizers (25 Sep): list every S2/S3 id of the business, not only the top one of identical records.
Rule 'dupfill': for each selected (id1, id2), add every twin of id2 that is a scored candidate of id1 and is not
selected for any S1. Reports twin prevalence, split rate, match rate of the added pairs and tsim F0.5 with the rule.
python ana/dups.py tag=fin2
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
tag = kv.get('tag', 'fin2')
cfg = json.load(open(f'{W}/dec_{tag}.json'))


def recs(sp):
    r = pl.concat([pl.read_parquet(f'{W}/n_{sp}_s{k}.parquet', columns=['id', 'name', 'addr', 'country']) for k in (2, 3)])
    r = r.select(enc('id').alias('id'), pl.concat_str([pl.col('name').fill_null(''), pl.col('addr').fill_null(''),
                                                       pl.col('country')], separator='\x1f').hash().alias('k'))
    return r.join(r.group_by('k').len('nk'), on='k')


def fill(sel, o, r):
    tw = r.filter(pl.col('nk') > 1)
    a = (sel.join(tw.rename({'id': 'id2'}), on='id2').select('id1', 'id2', 'k')
         .join(tw.rename({'id': 't'}).select('t', 'k'), on='k').filter(pl.col('t') != pl.col('id2'))
         .select('id1', pl.col('t').alias('id2')).unique())
    a = a.join(o.select('id1', 'id2'), on=['id1', 'id2'], how='semi').join(sel.select('id2'), on='id2', how='anti')
    return a, tw


o = pl.read_parquet(f'{W}/oof_{tag}.parquet').select('id1', 'id2', 'p')
sel = select(o, cfg)
r = recs('tsim')
add, tw = fill(sel, o, r)
s1 = pl.read_parquet(f'{W}/n_tsim_s1.parquet', columns=['id', 'country']).select(enc('id').alias('s1'), 'country')
gt = pl.read_parquet(gt_path('tsim')).select(enc('s1').alias('s1'), enc('id2').alias('id2'))
gp = gt.rename({'s1': 'id1'})
selt = sel.join(tw.rename({'id': 'id2'}), on='id2', how='semi')
print(f'tsim: S2/S3 records {r.height}, with an identical twin {tw.height} ({tw.height / r.height:.2%}); '
      f'selected links {sel.height}, of which to a record with twins {selt.height}', flush=True)
print(f'twins of selected records that are candidates of the same S1 but unselected: {add.height}; '
      f'true matches among them {add.join(gp, on=["id1", "id2"], how="semi").height}', flush=True)
b, n = score(sel, s1, gt), score(pl.concat([sel, add]).unique(), s1, gt)
print(f'F0.5 base {b["ALL"]:.6f} (US {b["US"]:.6f} IN {b["India"]:.6f}) -> dupfill {n["ALL"]:.6f} '
      f'(US {n["US"]:.6f} IN {n["India"]:.6f})  diff {n["ALL"] - b["ALL"]:+.6f}', flush=True)
# test: how many links the rule would add per country
pt = pl.read_parquet(f'{W}/pte_{tag}.parquet').select('id1', 'id2', 'p')
st = select(pt, cfg)
at, twt = fill(st, pt, recs('test'))
ct = pl.read_parquet(f'{W}/n_test_s1.parquet', columns=['id', 'country']).select(enc('id').alias('id1'), 'country')
print('test: records with twins', twt.height, '| links added per country:',
      dict(at.join(ct, on='id1').group_by('country').len().iter_rows()), flush=True)
