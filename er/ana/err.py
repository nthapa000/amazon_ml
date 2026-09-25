"""OOF error analysis: where the macro-F0.5 loss comes from.  python ana/err.py tag=s1 t=0.7"""
import sys
sys.path.insert(0, '/csehome/m25csa019/amazon_ml/er/src')
import polars as pl
from block import enc, gt_path
from decide import one2one, pick_thr, select
import json
from metric import macro_fbeta_arr

W = '/scratch/m25csa019/er_work'
kv = dict(a.split('=', 1) for a in sys.argv[1:])
o = pl.read_parquet(f'{W}/oof_{kv["tag"]}.parquet')
t = float(kv.get('t', 0.7))
cfg = json.load(open(f'{W}/dec_{kv["tag"]}.json'))
sel = select(o.select('id1', 'id2', 'p'), cfg)   # the version's actual decision rule
SP = kv.get('split', 'train')
gt = pl.read_parquet(gt_path(SP)).select(enc('s1').alias('id1'), enc('id2').alias('id2'))
s1 = pl.read_parquet(f'{W}/n_{SP}_s1.parquet', columns=['id', 'country']).select(enc('id').alias('id1'), 'country')
cand = o.select('id1', 'id2')
tp = sel.join(gt, on=['id1', 'id2'], how='semi')
fp = sel.join(gt, on=['id1', 'id2'], how='anti')
fn = gt.join(sel, on=['id1', 'id2'], how='anti')
fn = fn.with_columns(inc=pl.lit(True)).join(cand.with_columns(c=pl.lit(1)), on=['id1', 'id2'], how='left') \
    .with_columns(blocked=pl.col('c').is_null())
owner = gt.rename({'id1': 'own'})
fp = fp.join(owner, on='id2', how='left').with_columns(kind=pl.when(pl.col('own').is_null()).then(pl.lit('distractor'))
                                                        .otherwise(pl.lit('other_owner')))
per = (s1.join(gt.group_by('id1').len('nt'), on='id1', how='left')
       .join(sel.group_by('id1').len('np'), on='id1', how='left')
       .join(tp.group_by('id1').len('ntp'), on='id1', how='left').fill_null(0))
f, fa = macro_fbeta_arr(per['nt'], per['np'], per['ntp'])
per = per.with_columns(loss=1 - pl.Series(fa))
print('F0.5', f, 'total loss', per['loss'].sum())
per = per.with_columns(cat=pl.when(pl.col('nt') == 0).then(pl.lit('singleton_fp'))
                       .when(pl.col('np') == 0).then(pl.lit('pred_empty'))
                       .when(pl.col('ntp') == 0).then(pl.lit('all_wrong'))
                       .when(pl.col('np') > pl.col('ntp')).then(pl.lit('has_fp'))
                       .otherwise(pl.lit('only_fn')))
print(per.group_by('cat', 'country').agg(pl.len(), pl.col('loss').sum()).sort('loss', descending=True))
print('FP kinds', fp.group_by('kind').len())
ns = per.filter(pl.col('nt') == 0).height
print('true singletons', ns, 'predicted empty correctly', per.filter((pl.col('nt') == 0) & (pl.col('np') == 0)).height,
      '| entities with matches predicted empty', per.filter((pl.col('nt') > 0) & (pl.col('np') == 0)).height)
print('precision', tp.height / max(sel.height, 1), 'recall', tp.height / gt.height)
print('FN blocked vs thresholded', fn.group_by('blocked').len())
# p distribution of FN that were candidates
print(o.join(fn.filter(~pl.col('blocked')).select('id1', 'id2'), on=['id1', 'id2']).select(pl.col('p').qcut(10, include_breaks=True).struct.field('breakpoint').unique().sort()))
fp.write_parquet(f'{W}/ana_fp_{kv["tag"]}.parquet')
fn.write_parquet(f'{W}/ana_fn_{kv["tag"]}.parquet')
