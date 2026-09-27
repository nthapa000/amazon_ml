"""Rows of two submissions that differ, per S1 country (e.g. a France-only change must leave US/India rows identical).
python ana/diff_country.py a=<dir with matching_results.tsv> b=<dir>"""
import os
import sys

import polars as pl

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')
kv = dict(x.split('=', 1) for x in sys.argv[1:])
r = lambda d: pl.read_csv(f'{d}/matching_results.tsv', separator='\t', infer_schema_length=0, quote_char=None).fill_null('')
a, b = r(kv['a']), r(kv['b'])
k, v = a.columns
c = pl.read_parquet(f'{W}/n_test_s1.parquet', columns=['id', 'country']).rename({'id': k})
m = a.join(b, on=k, suffix='_b').join(c, on=k, how='left')
print(a.height, b.height, m.height)
print(m.group_by('country').agg(changed=(pl.col(v) != pl.col(v + '_b')).sum(), rows=pl.len()).sort('country'))
