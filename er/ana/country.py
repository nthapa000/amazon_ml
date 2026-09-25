import sys
sys.path.insert(0, '/csehome/m25csa019/amazon_ml/er/src')
import polars as pl
from block import enc
W = '/scratch/m25csa019/er_work'
tag = sys.argv[1]
s1 = pl.read_parquet(f'{W}/n_test_s1.parquet', columns=['id', 'country']).select(enc('id').alias('id1'), 'country')
p = pl.read_parquet(f'{W}/pte_{tag}.parquet').join(s1, on='id1')
best = p.group_by('id1', 'country').agg(pl.col('p').max().alias('mx'), (pl.col('p') > 0.5).sum().alias('n05'), pl.len().alias('nc'))
print(best.group_by('country').agg(pl.col('mx').mean(), (pl.col('mx') > 0.9).mean().alias('mx>0.9'), (pl.col('mx') < 0.3).mean().alias('mx<0.3'),
                                   pl.col('n05').mean(), pl.col('nc').mean()))
print(p.group_by('country').agg(((pl.col('p') > 0.1) & (pl.col('p') < 0.9)).mean().alias('uncertain_frac'), pl.len()))
o = pl.read_parquet(f'{W}/oof_{tag}.parquet')
s1t = pl.read_parquet(f'{W}/n_tsim_s1.parquet', columns=['id', 'country']).select(enc('id').alias('id1'), 'country')
o = o.join(s1t, on='id1')
b = o.group_by('id1', 'country').agg(pl.col('p').max().alias('mx'), (pl.col('p') > 0.5).sum().alias('n05'), pl.len().alias('nc'))
print('OOF tsim'); print(b.group_by('country').agg(pl.col('mx').mean(), (pl.col('mx') > 0.9).mean().alias('mx>0.9'), (pl.col('mx') < 0.3).mean().alias('mx<0.3'), pl.col('n05').mean(), pl.col('nc').mean()))
print(o.group_by('country').agg(((pl.col('p') > 0.1) & (pl.col('p') < 0.9)).mean().alias('uncertain_frac'), pl.len()))
