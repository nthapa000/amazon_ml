"""Sample GT pairs missed by uncapped blocking (cand0_{split}_{ctry}) and print normalized fields."""
import os, sys
import polars as pl
sys.path.insert(0, os.path.dirname(__file__))
from block import enc
W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')
pat = sys.argv[1] if len(sys.argv) > 1 else 'cand0_train_*'
pr = pl.scan_parquet(f'{W}/{pat}.parquet').select('id1', 'id2')
gt = pl.read_parquet(f'{W}/gt_pairs.parquet').select(enc('s1').alias('id1'), enc('id2').alias('id2'), pl.col('s1').alias('s1s'), pl.col('id2').alias('id2s'))
miss = gt.lazy().join(pr, on=['id1', 'id2'], how='anti').collect()
print('missed', miss.height, flush=True)
cols = ['id', 'name', 'country', 'core', 'ad', 'hn', 'state']
s1 = pl.read_parquet(f'{W}/n_train_s1.parquet', columns=cols)
s23 = pl.concat([pl.read_parquet(f'{W}/n_train_s{k}.parquet', columns=cols) for k in (2, 3)])
m = miss.join(s1.rename({'id': 's1s'}), on='s1s').join(s23.rename({'id': 'id2s'}), on='id2s', suffix='2')
m = m.with_columns(e2=pl.col('ad2') == '', same_core=pl.col('core') == pl.col('core2'),
                   share_tok=pl.col('core').str.split(' ').list.set_intersection(pl.col('core2').str.split(' ')).list.len() > 0)
print(m.group_by('country').agg(pl.len(), pl.col('e2').mean(), pl.col('same_core').mean(), pl.col('share_tok').mean()))
# how common is the S1 core name?
cc = s1.group_by('country', 'core').len('ncore')
m = m.join(cc, on=['country', 'core'], how='left')
print(m.group_by('country').agg(pl.col('ncore').median(), (pl.col('ncore') > 60).mean().alias('gt60')))
with pl.Config(tbl_rows=60, fmt_str_lengths=60, tbl_width_chars=250):
    for c in ('US', 'India'):
        print(m.filter(pl.col('country') == c).sample(25, seed=2).select('name', 'name2', 'ncore', 'ad', 'ad2'))
