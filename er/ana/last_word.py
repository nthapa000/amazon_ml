"""Unsupervised check (no labels): does the last-word signature of true matches also appear in the unseen country?
In labelled US/India matches a whole-word change is almost always the LAST word, replaced by one of {center, services, service,
partners} (ana/name_noise.py). Here: per split / country / source, the share of core names whose last word is one of them,
and the top last words. S1 is the clean side, so a much higher share on S2/S3 means the same noise was applied there.
python ana/last_word.py     (ER_WORK = work dir)"""
import os
from collections import Counter

import polars as pl

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')
G = ['center', 'services', 'service', 'partners']
for sp in ('train', 'test'):
    for src in (1, 2, 3):
        d = pl.read_parquet(f'{W}/n_{sp}_s{src}.parquet', columns=['country', 'core']).with_columns(
            last=pl.col('core').fill_null('').str.split(' ').list.last().fill_null(''))
        for c in sorted(d['country'].unique().to_list()):
            x = d.filter(pl.col('country') == c)
            share = x.filter(pl.col('last').is_in(G)).height / x.height
            top = Counter(x['last'].to_list()).most_common(6)
            print(f'{sp} S{src} {c:7s} n={x.height:8d} last word in {G}: {share:.4f} | top last words {top}', flush=True)
