"""Does a missing `state` on the S2/S3 side actually cost match quality? (labelled US/India tsim, so we can check for real
before trusting the same logic on France). Candidate pairs (p>=0.001) split by whether the S2/S3 side has a state, and by
whether S1/S2 state AGREES (when both present) -- match rate and mean p in the uncertain band.
python ana/state_value.py     (ER_WORK = work dir)"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import polars as pl  # noqa: E402
from block import enc  # noqa: E402

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')
p = pl.read_parquet(f'{W}/oof_b1ka.parquet').filter(pl.col('p') >= 0.001)
s1 = pl.read_parquet(f'{W}/n_tsim_s1.parquet', columns=['id', 'country', 'state']).select(enc('id').alias('id1'), 'country', pl.col('state').alias('st1'))
s23 = pl.concat([pl.read_parquet(f'{W}/n_tsim_s{k}.parquet', columns=['id', 'state']) for k in (2, 3)]).select(enc('id').alias('id2'), pl.col('state').alias('st2'))
d = p.join(s1, on='id1').join(s23, on='id2').with_columns(pl.col('st1').fill_null(''), pl.col('st2').fill_null(''))
d = d.with_columns(cls=pl.when(pl.col('st2') == '').then(pl.lit('s2_missing'))
                    .when(pl.col('st1') == pl.col('st2')).then(pl.lit('agree'))
                    .otherwise(pl.lit('disagree')))
band = (pl.col('p') > 0.1) & (pl.col('p') < 0.9)
with pl.Config(tbl_rows=20):
    print(d.group_by('country', 'cls').agg(pl.len().alias('pairs'), pl.col('y').mean().alias('match_rate'),
          band.sum().alias('n_band'), pl.col('y').filter(band).mean().alias('match_in_band')).sort('country', 'cls'))
