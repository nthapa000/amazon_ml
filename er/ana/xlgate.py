"""L1 gate (plan (1).md IV.2): are native-script India records under-served by the current model?
True pairs of India S1 on tsim, split by the S2/S3 record's script: latin (nind == 0), native with every token in the
learned dictionary (nunk == 0), native with unknown tokens (nunk > 0). Reports share of true pairs, recall of the
final selection, blocked (not a stage-2 candidate) and wrong-owner rates, plus false links by the same split.

python ana/xlgate.py tag=fin5s
"""
import os
import sys
import json

import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from block import enc, gt_path  # noqa: E402
from decide import select  # noqa: E402

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')
kv = dict(a.split('=', 1) for a in sys.argv[1:])
tag = kv.get('tag', 'fin5s')
cfg = json.load(open(f'{W}/dec_{tag}.json'))
o = pl.read_parquet(f'{W}/oof_{tag}.parquet', columns=['id1', 'id2', 'p'])
sel = select(o, cfg).with_columns(s=pl.lit(1))
s1 = pl.read_parquet(f'{W}/n_tsim_s1.parquet', columns=['id', 'country']).select(enc('id').alias('id1'), 'country')
rec = pl.concat([pl.read_parquet(f'{W}/n_tsim_s{k}.parquet', columns=['id', 'nind', 'nunk']) for k in (2, 3)])
rec = rec.select(enc('id').alias('id2'), grp=pl.when(pl.col('nind').fill_null(0) == 0).then(pl.lit('latin'))
                 .when(pl.col('nunk').fill_null(0) == 0).then(pl.lit('native_known')).otherwise(pl.lit('native_unk')))
gt = pl.read_parquet(gt_path('tsim')).select(enc('s1').alias('id1'), enc('id2').alias('id2'))
t = (gt.join(s1, on='id1').join(rec, on='id2', how='left')
     .join(sel.select('id1', 'id2', 's'), on=['id1', 'id2'], how='left')
     .join(o.select('id1', 'id2', c=pl.lit(1)), on=['id1', 'id2'], how='left')
     .join(sel.select('id2', other=pl.lit(1)), on='id2', how='left')
     .with_columns(pl.col('s').fill_null(0), pl.col('c').fill_null(0), pl.col('other').fill_null(0)))
with pl.Config(tbl_rows=20, tbl_width_chars=200, float_precision=4):
    print(t.group_by('country', 'grp').agg(
        true_pairs=pl.len(), recall=pl.col('s').mean(), blocked=(pl.col('c') == 0).mean(),
        missed_in_cands=((pl.col('c') == 1) & (pl.col('s') == 0)).mean(),
        wrong_owner=((pl.col('s') == 0) & (pl.col('other') == 1)).mean()).sort('country', 'grp'))
    fp = sel.join(gt.with_columns(y=pl.lit(1)), on=['id1', 'id2'], how='left').filter(pl.col('y').is_null())
    fp = fp.join(s1, on='id1').join(rec, on='id2', how='left')
    ln = sel.join(s1, on='id1').join(rec, on='id2', how='left')
    print(ln.group_by('country', 'grp').len('links').join(fp.group_by('country', 'grp').len('false_links'),
                                                          on=['country', 'grp'], how='left')
          .with_columns(fp_rate=pl.col('false_links') / pl.col('links')).sort('country', 'grp'))
