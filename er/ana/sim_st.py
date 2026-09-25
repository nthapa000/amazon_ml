"""France-risk simulation: how much does F0.5 drop if st_eq is always 'unknown' (-1), as it is for every French pair?
Uses the w1 stage-1 fold models on their own held-out folds (OOF), on a 25% S1 sample of tsim.
python ana/sim_st.py
"""
import glob
import sys

import numpy as np
import polars as pl
import lightgbm as lgb

sys.path.insert(0, '/csehome/m25csa019/amazon_ml/er/src')
from block import enc, gt_path, FAM  # noqa: E402
from decide import select, score  # noqa: E402

W = '/scratch/m25csa019/er_work'
keep = lambda c: (pl.col(c).hash(seed=99) % 4 == 0)      # 25% of S1
fs = sorted(glob.glob(f'{W}/f_cand_tsim_kx/*.parquet'))
d = pl.concat([pl.read_parquet(f).filter(keep('id1')) for f in fs], how='vertical_relaxed')
for e in ('xf_tsim_kx', 'f2_tsim_kx', 'sib_tsim'):
    d = d.join(pl.read_parquet(f'{W}/{e}.parquet').filter(keep('id1')), on=['id1', 'id2'], how='left')
d = d.with_columns([((pl.col('kbits') & v[0]) > 0).cast(pl.Int8).alias('k_' + f) for f, v in FAM.items()])
d = d.with_columns(fold=(pl.col('id1').hash(seed=7) % 5).cast(pl.Int8))
m0 = lgb.Booster(model_file=f'{W}/lgb_w1_0.txt')
feats = m0.feature_name()
print('rows', d.height, 'st_eq known share', (d['st_eq'] != -1).mean(), flush=True)
p0 = np.zeros(d.height, np.float32)
p1 = np.zeros(d.height, np.float32)
fold = d['fold'].to_numpy()
X = d.select(feats).to_numpy().astype(np.float32)
ist = feats.index('st_eq')
for k in range(5):
    m = lgb.Booster(model_file=f'{W}/lgb_w1_{k}.txt')
    idx = fold == k
    p0[idx] = m.predict(X[idx])
    Xk = X[idx].copy()
    Xk[:, ist] = -1.0
    p1[idx] = m.predict(Xk)
s1 = pl.read_parquet(f'{W}/n_tsim_s1.parquet', columns=['id', 'country']).select(enc('id').alias('s1'), 'country') \
    .filter(keep('s1'))
gt = pl.read_parquet(gt_path('tsim')).select(enc('s1').alias('s1'), enc('id2').alias('id2')).filter(keep('s1'))
cfg = dict(method='ef', margin=0.2, m=0.0, a=1.0)
base = d.select('id1', 'id2')
for name, p in (('original', p0), ('st_eq forced -1', p1)):
    r = score(select(base.with_columns(p=pl.Series(p)), cfg), s1, gt)
    print(f'{name:18s} F0.5 {r["ALL"]:.5f}  US {r["US"]:.5f}  India {r["India"]:.5f}  pred_empty {r["pred_empty"]:.4f}',
          flush=True)
print('mean p on true pairs: original', float(p0[d['y'].to_numpy() == 1].mean()) if 'y' in d.columns else '',
      flush=True)
