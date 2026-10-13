"""Hard-negative features (polars, vectorised), motivated by OOF error analysis:
- distractors often copy the address and change a short initials token (GAA -> GIA, BP -> TP, WX -> WF)
- distractors with the same name often have a different house number (113 vs 120), while true-match
  noise on house numbers is mostly truncation (3029 -> 029, 9652 -> 965)

python feats2.py split=tsim cand=cand_tsim_k100 out=f2_tsim_k100  -> W/{out}.parquet
"""
import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
from block import enc  # noqa: E402

W = os.environ.get('ER_WORK', 'work')


def prep(d, sfx):
    """Per-record inputs: short (initials) tokens, first token, first house number, token count."""
    toks = pl.col('core').str.split(' ')
    short = toks.list.eval(pl.element().filter((pl.element().str.len_chars() <= 3)
                                               & pl.element().str.contains(r'^[a-z]+$')))
    first_num = pl.col('hn').str.extract(r'(\d+)', 1)
    return d.select(pl.col('id').alias('id' + sfx),
                    short.alias('sh' + sfx),
                    toks.list.first().alias('ft' + sfx),
                    first_num.alias('h' + sfx),
                    pl.col('hn').alias('hn' + sfx),
                    toks.list.len().cast(pl.Int32).alias('nt' + sfx))


def main(split, cand, out=None):
    """Initials and house-number features of a candidate set."""
    c = ['id', 'core', 'hn']
    s1 = prep(pl.read_parquet(f'{W}/n_{split}_s1.parquet', columns=c).with_columns(id=enc('id')), '1')
    s23 = prep(pl.concat([pl.read_parquet(f'{W}/n_{split}_s{k}.parquet', columns=c) for k in (2, 3)])
               .with_columns(id=enc('id')), '2')
    pr = pl.read_parquet(f'{W}/{cand}.parquet', columns=['id1', 'id2'])
    d = pr.join(s1, on='id1', how='left').join(s23, on='id2', how='left')
    has = (pl.col('sh1').list.len() > 0) & (pl.col('sh2').list.len() > 0)
    h1, h2 = pl.col('h1'), pl.col('h2')
    both_h = h1.is_not_null() & h2.is_not_null() & (h1 != '') & (h2 != '')
    n1, n2 = h1.cast(pl.Int64, strict=False), h2.cast(pl.Int64, strict=False)
    d = d.select(
        'id1', 'id2',
        # short-token (initials) agreement
        pl.when(has).then(pl.col('sh1').list.set_intersection('sh2').list.len()
                          / pl.max_horizontal(pl.col('sh1').list.len(), pl.col('sh2').list.len()))
        .otherwise(-1.0).alias('sh_jac'),
        pl.when(has).then((pl.col('sh1').list.first() != pl.col('sh2').list.first()).cast(pl.Float32))
        .otherwise(-1.0).alias('sh_first_diff'),
        (pl.col('ft1') == pl.col('ft2')).cast(pl.Float32).alias('ft_eq'),
        (pl.col('nt1') - pl.col('nt2')).alias('nt_diff'),
        # house number: numeric distance and truncation pattern
        pl.when(both_h).then((n1 - n2).abs().cast(pl.Float32).log1p()).otherwise(-1.0).alias('hn_ldiff'),
        pl.when(both_h).then((h1.str.contains(h2, literal=True) | h2.str.contains(h1, literal=True))
                             .cast(pl.Float32)).otherwise(-1.0).alias('hn_substr'),
        pl.when(both_h).then((h1.str.len_chars().cast(pl.Int32) - h2.str.len_chars().cast(pl.Int32)).cast(pl.Float32)).otherwise(-99.0)
        .alias('hn_lendiff'),
    )
    d.write_parquet(f'{W}/{out or "f2_" + split}.parquet')
    print(d.describe())


if __name__ == '__main__':
    kv = dict(a.split('=', 1) for a in sys.argv[1:])
    main(kv['split'], kv['cand'], kv.get('out'))
