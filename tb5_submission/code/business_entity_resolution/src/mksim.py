"""Build a test-like training split 'tsim': drop a fraction of train S1 entities so their S2/S3 records become
orphans (no owner in S1). drop=0.19 matches the test ratio of S2+S3 records per S1 (5.76 vs 4.68 in train).
python mksim.py drop=0.19  -> n_tsim_s{1,2,3}.parquet, gt_pairs_tsim.parquet
"""
import os
import sys

import polars as pl

W = os.environ.get('ER_WORK', 'work')
kv = dict(a.split('=', 1) for a in sys.argv[1:])
drop = float(kv.get('drop', 0.19))
key = pl.col('id').str.slice(3).cast(pl.Int64).hash(seed=2026) % 10000
s1 = pl.read_parquet(f'{W}/n_train_s1.parquet')
keep = s1.filter(key >= int(drop * 10000))
keep.write_parquet(f'{W}/n_tsim_s1.parquet')
for k in (2, 3):
    dst = f'{W}/n_tsim_s{k}.parquet'
    if not os.path.exists(dst):
        os.symlink(f'{W}/n_train_s{k}.parquet', dst)
gt = pl.read_parquet(f'{W}/gt_pairs.parquet').join(keep.select(pl.col('id').alias('s1')), on='s1', how='semi')
gt.write_parquet(f'{W}/gt_pairs_tsim.parquet')
n23 = sum(pl.scan_parquet(f'{W}/n_train_s{k}.parquet').select(pl.len()).collect().item() for k in (2, 3))
print('kept S1', keep.height, 'of', s1.height, 'S23/S1', n23 / keep.height, 'links/S1', gt.height / keep.height)
