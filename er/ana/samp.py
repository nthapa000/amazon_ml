import sys
sys.path.insert(0, '/csehome/m25csa019/amazon_ml/er/src')
import polars as pl
from block import enc
W = '/scratch/m25csa019/er_work'
cols = ['id', 'name', 'addr', 'country']
s1 = pl.read_parquet(f'{W}/n_train_s1.parquet', columns=cols).with_columns(id=enc('id'))
s23 = pl.concat([pl.read_parquet(f'{W}/n_train_s{k}.parquet', columns=cols) for k in (2, 3)]).with_columns(id=enc('id'))
o = pl.read_parquet(f'{W}/oof_s1.parquet')
fn = pl.read_parquet(f'{W}/ana_fn_s1.parquet').filter(~pl.col('blocked')).join(o, on=['id1', 'id2'])
fp = pl.read_parquet(f'{W}/ana_fp_s1.parquet').join(o, on=['id1', 'id2'])
def show(d, n, title, extra=None):
    d = d.sample(n, seed=5).join(s1.rename({'id': 'id1'}), on='id1').join(s23.rename({'id': 'id2'}), on='id2', suffix='2')
    if extra is not None:
        d = d.join(s1.rename({'id': 'own', 'name': 'oname', 'addr': 'oaddr', 'country': 'oc'}), on='own', how='left')
    print('=====', title)
    with pl.Config(tbl_rows=40, fmt_str_lengths=50, tbl_width_chars=300):
        c = ['p', 'name', 'name2', 'addr', 'addr2'] + (['oname', 'oaddr'] if extra is not None else [])
        print(d.select(c))
show(fn.filter(pl.col('p') > 0.3), 25, 'FN with p>0.3 (near miss)')
show(fn.filter(pl.col('p') < 0.1), 20, 'FN with p<0.1')
show(fp.filter(pl.col('kind') == 'other_owner'), 20, 'FP other owner', extra=True)
show(fp.filter(pl.col('kind') == 'distractor'), 20, 'FP distractor')
