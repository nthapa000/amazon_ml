"""Find the words the noise process inserts as the LAST word of a name, without labels.
Noise-inserted words are over-represented on the noisy side (S2/S3) relative to the clean side (S1) within a country:
excess(w) = share of S2/S3 names ending in w - share of S1 names ending in w.
Validation: on labelled US/India (train) the top-excess words must be the words seen in true matches (center, services, service,
partners; ana/name_noise.py). Then the same statistic on the unseen country (test files only, no labels).
python ana/noise_words.py [top=12]     (ER_WORK = work dir)"""
import os
import sys

import polars as pl

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')
kv = dict(a.split('=', 1) for a in sys.argv[1:])
top = int(kv.get('top', 12))


def last(sp, srcs):
    d = pl.concat([pl.read_parquet(f'{W}/n_{sp}_s{k}.parquet', columns=['country', 'core']) for k in srcs])
    return d.with_columns(w=pl.col('core').fill_null('').str.split(' ').list.last().fill_null('')).filter(pl.col('w') != '')


for sp in ('train', 'test'):
    a, b = last(sp, (1,)), last(sp, (2, 3))
    for c in sorted(a['country'].unique().to_list()):
        f1 = a.filter(pl.col('country') == c).group_by('w').len('n1').with_columns(f1=pl.col('n1') / pl.col('n1').sum())
        f2 = b.filter(pl.col('country') == c).group_by('w').len('n2').with_columns(f2=pl.col('n2') / pl.col('n2').sum())
        e = f2.join(f1, on='w', how='left').with_columns(pl.col('f1').fill_null(0), ex=pl.col('f2') - pl.col('f1').fill_null(0)).sort('ex', descending=True)
        print(f'== {sp} {c}: S1 names {a.filter(pl.col("country") == c).height}, S2/S3 names {b.filter(pl.col("country") == c).height}')
        for w, s2, s1, x in e.head(top).select('w', 'f2', 'f1', 'ex').iter_rows():
            print(f'   {w:16s} S2/S3 {s2:.4f}  S1 {s1:.4f}  excess {x:+.4f}')
