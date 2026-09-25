"""Name-truncation / prefix-ambiguity features (from the c2 worst-error review).

Pattern: many missed links are S2/S3 records whose name keeps only the first words of the S1 name
("Green Creative Dynamics" -> "Green Creative"), often with no address. Whether such a record is safe to link
depends on how many S1 businesses start with that prefix.

- p_is   : the S2/S3 core name is a word-prefix of the S1 core name (1 = proper prefix, 2 = equal, 0 = no)
- p_rev  : the S1 core name is a word-prefix of the S2/S3 core name (S2/S3 adds words)
- p_n1   : #S1 in the country whose core name starts with the S2/S3 core name (word-prefix, incl. equal)
- p_nc   : #candidate S1 of this S2/S3 record (in the given candidate set) for which p_is > 0
- p_len  : #words of the S2/S3 core name

python xf2.py split=tsim cand=cand_tsim_kx out=xf2_tsim
"""
import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
from block import enc  # noqa: E402

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')


def main(kv):
    split, cand = kv['split'], kv['cand']
    c = ['id', 'country', 'core']
    s1 = pl.read_parquet(f'{W}/n_{split}_s1.parquet', columns=c).with_columns(id=enc('id'))
    s23 = pl.concat([pl.read_parquet(f'{W}/n_{split}_s{k}.parquet', columns=c) for k in (2, 3)]).with_columns(id=enc('id'))
    # all word-prefixes of every S1 core name -> counts per (country, prefix)
    t = s1.select('country', pl.col('core').str.split(' ').alias('w'))
    t = t.with_columns(k=pl.int_ranges(1, pl.col('w').list.len() + 1)).explode('k').drop_nulls('k')
    pref = (t.select('country', pl.col('w').list.head(pl.col('k')).list.join(' ').alias('pfx'))
            .group_by('country', 'pfx').len('p_n1'))
    x = s23.join(pref.rename({'pfx': 'core'}), on=['country', 'core'], how='left').with_columns(pl.col('p_n1').fill_null(0))
    pr = pl.read_parquet(f'{W}/{cand}.parquet', columns=['id1', 'id2'])
    d = (pr.join(s1.select(pl.col('id').alias('id1'), pl.col('core').alias('c1')), on='id1', how='left')
         .join(x.select(pl.col('id').alias('id2'), pl.col('core').alias('c2'), 'p_n1'), on='id2', how='left')
         .with_columns(pl.col('c1').fill_null(''), pl.col('c2').fill_null('')))
    eq = pl.col('c1') == pl.col('c2')
    d = d.with_columns(
        p_is=pl.when((pl.col('c2') != '') & eq).then(2).when(
            (pl.col('c2') != '') & pl.col('c1').str.starts_with(pl.col('c2') + ' ')).then(1).otherwise(0).cast(pl.Int8),
        p_rev=((pl.col('c1') != '') & ~eq & pl.col('c2').str.starts_with(pl.col('c1') + ' ')).cast(pl.Int8),
        p_len=pl.col('c2').str.split(' ').list.len().cast(pl.Int16))
    d = d.with_columns(p_nc=(pl.col('p_is') > 0).sum().over('id2').cast(pl.Int16))
    d.select('id1', 'id2', 'p_is', 'p_rev', 'p_n1', 'p_nc', 'p_len').write_parquet(f'{W}/{kv["out"]}.parquet')
    print(d.group_by('p_is').agg(pl.len(), pl.col('p_n1').median(), pl.col('p_nc').mean()).sort('p_is'))


if __name__ == '__main__':
    main(dict(a.split('=', 1) for a in sys.argv[1:]))
