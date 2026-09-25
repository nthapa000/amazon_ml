import sys; sys.path.insert(0,'/csehome/m25csa019/amazon_ml/er/src')
import polars as pl
from block import enc
W='/scratch/m25csa019/er_work'
x=pl.read_parquet(f'{W}/xf_train.parquet')
gt=pl.read_parquet(f'{W}/gt_pairs.parquet').select(enc('s1').alias('id1'),enc('id2').alias('id2'),y=pl.lit(1))
f=pl.read_parquet(f'{W}/f_cand_train/p_00.parquet',columns=['id1','id2','co_ts','ad_ts','hn_inter','hn_conf'])
d=f.join(x,on=['id1','id2']).join(gt,on=['id1','id2'],how='left').with_columns(pl.col('y').fill_null(0))
d=d.with_columns(invented=(pl.col('v2_max') if 'v2_max' in d.columns else pl.col('v2_mean'))==0, addr_hi=pl.col('ad_ts')>=90, name_lo=pl.col('co_ts')<40)
print(d.group_by('name_lo','addr_hi','invented').agg(pl.len(), pl.col('y').mean()).sort('name_lo','addr_hi','invented'))
print(d.filter(pl.col('name_lo')&pl.col('addr_hi')).group_by((pl.col('v2_mean')*2).round(0)/2).agg(pl.len(),pl.col('y').mean()).sort('v2_mean'))
