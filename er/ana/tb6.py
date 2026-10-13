"""tb6 = tb5 with France 0.5 kcxQ + 0.25 fin5s + 0.25 kfx; tb5 = tb3 with our France member swapped: fin6s (ceqf drop-in) -> fin5s (plain ceq; LB showed the ceqf drop-in hurts France).
Fixed weights (no search): US/India 0.7 kcxQ + 0.3 fin5s + 0 kfx (= tb3's tuned weights; fin5s == fin6s on tsim and US/India test),
France equal thirds. Writes pte_tb6 / oof_tb6 / dec_tb6 / blend_tb6 in ER_WORK. No src/ edits."""
import os, sys, json, shutil
import numpy as np, polars as pl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wblend import load, mix, W, enc, unseen  # noqa: E402
tags = ['kcxQ', 'fin5s', 'kfx']
best, wu = np.array([0.7, 0.3, 0.0]), np.array([0.5, 0.25, 0.25])
tids, TL, TM = load('pte', tags)
t1 = pl.read_parquet(f'{W}/n_test_s1.parquet', columns=['id', 'country'])
cs = unseen(t1)
un_ids = t1.filter(pl.col('country').is_in(cs)).select(enc('id').alias('id1'))
is_un = tids.join(un_ids.with_columns(u=pl.lit(True)), on='id1', how='left')['u'].fill_null(False).to_numpy()
p = np.where(is_un, mix(TL, TM, wu), mix(TL, TM, best))
tids.with_columns(p=pl.Series(p, dtype=pl.Float32)).write_parquet(f'{W}/pte_tb6.parquet')
shutil.copy(f'{W}/oof_tb3.parquet', f'{W}/oof_tb6.parquet')
shutil.copy(f'{W}/dec_tb3.json', f'{W}/dec_tb6.json')
json.dump(dict(tags=tags, weights=best.tolist(), unseen_weights=wu.tolist(), unseen_countries=cs), open(f'{W}/blend_tb6.json', 'w'))
print(f'test rows {tids.height}, unseen-country rows {int(is_un.sum())}, unseen {cs}', flush=True)
