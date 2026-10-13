"""Blocking v2: compound keys (name x address words, address word pairs, full name, ...) joined per country,
IDF-weighted key score, then top-K per S1 and top-k per S2/S3 -> candidate set.

python block.py gen tsim       -> W/cand0_{split}_<country>.parquet (all key pairs with score), W/cand_{split}.parquet
                                  (capped at K1/K2) + recall report when the split has labels
python block.py report tsim [cand_tsim_k100]
Ids are encoded as int64: src * 1e10 + number (S1-123 -> 10000000123).
"""
import os
import sys
import time

import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
from metric import macro_fbeta_arr  # noqa: E402

W = os.environ.get('ER_WORK', 'work')
K1 = int(os.environ.get('K1', 40))      # max candidates per S1
K2 = int(os.environ.get('K2', 6))       # max S1 per S2/S3 record
# family: (bit, cap S1 side, cap S2/3 side)
FAM = {'tok': (1, 15, 150), 'pair': (2, 40, 400), 'tokst': (4, 25, 250), 'cat': (8, 40, 400),
       'full': (16, 60, 600), 'nw': (32, 12, 120), 'ww': (64, 12, 120), 'hw': (128, 12, 120),
       'fw': (256, 12, 120), 'code': (512, 12, 120), 'sub': (1024, 25, 250)}


def gt_path(split):
    """Ground-truth pair file of a split (train: all labels; tsim: labels of the kept S1)."""
    return f'{W}/gt_pairs.parquet' if split == 'train' else f'{W}/gt_pairs_{split}.parquet'


def enc(col):
    """Entity id 'Sk-<n>' -> int64 k * 1e10 + n (compact join key)."""
    return (pl.col(col).str.slice(1, 1).cast(pl.Int64) * 10_000_000_000 + pl.col(col).str.slice(3).cast(pl.Int64))


def dec(col):
    """Inverse of enc: int64 -> 'Sk-<n>'."""
    return (pl.lit('S') + (pl.col(col) // 10_000_000_000).cast(pl.String) + pl.lit('-')
            + (pl.col(col) % 10_000_000_000).cast(pl.String))


def load(split, ctry):
    """Normalized records of one country of a split: S1 on side 0, S2/S3 on side 1, encoded ids."""
    cols = ['id', 'country', 'core', 'dom', 'ad', 'hn', 'state']
    fr = []
    for k in (1, 2, 3):
        d = pl.read_parquet(f'{W}/n_{split}_s{k}.parquet', columns=cols).filter(pl.col('country') == ctry)
        fr.append(d.with_columns(side=pl.lit(int(k > 1), pl.Int8)))
    return pl.concat(fr).with_columns(id=enc('id'))


def keys(rec):
    """-> DataFrame(id:i64, side:i8, key:u64, fam:str)"""
    n = rec.height
    adw = rec.select('id', pl.col('ad').str.replace_all(r'[^a-z ]', ' ').str.split(' ').list.eval(
        pl.element().filter(pl.element().str.len_chars() >= 3)).list.unique().alias('w')).explode('w').drop_nulls()
    wdf = adw.group_by('w').len('df')
    # drop generic address words (>0.3% of records) from compound keys
    rare = wdf.filter(pl.col('df') <= max(50, int(0.003 * n)))
    adall = adw.join(wdf, on='w').sort(['id', 'df']).group_by('id', maintain_order=True).head(10)
    adw = adw.join(rare, on='w').sort(['id', 'df']).group_by('id', maintain_order=True).head(8)
    base = rec.select('id', 'side', 'state', 'dom',
                      pl.col('core').str.split(' ').list.eval(pl.element().filter(pl.element().str.len_chars() >= 2))
                      .list.unique(maintain_order=True).alias('ct'),
                      pl.col('hn').str.split(' ').list.eval(pl.element().filter(
                          (pl.element().str.len_chars() >= 2) | pl.element().str.contains('[a-z]'))).alias('hl0'),
                      pl.col('ad').str.extract_all(r'[a-z0-9/\-]*\d[a-z0-9/\-]*').list.eval(pl.element().filter(
                          (pl.element().str.len_chars() >= 3) & (pl.element().str.contains(r'[a-z/\-]')
                                                                 | (pl.element().str.len_chars() >= 4)))
                      ).list.unique().alias('codes'))
    # house numbers plus digit-only variants (700b -> 700)
    base = base.with_columns(hl=pl.concat_list('hl0', pl.col('hl0').list.eval(
        pl.element().str.replace_all('[a-z]', '')).list.eval(pl.element().filter(pl.element().str.len_chars() >= 2)))
        .list.unique())
    sd = base.select('id', 'side')
    out = []

    def add(df, fam, expr):
        """Append one key family: one hashed key per (record, key value)."""
        out.append(df.select('id', expr.hash(seed=FAM[fam][0]).alias('key'), pl.lit(fam).alias('fam')))

    t = base.select('id', pl.col('ct').list.head(5).alias('t')).explode('t').drop_nulls()
    add(t, 'tok', pl.col('t'))
    add(t.join(base.select('id', 'state'), on='id').filter(pl.col('state') != ''), 'tokst',
        pl.col('t') + '|' + pl.col('state'))
    p = base.select('id', pl.col('ct').list.head(5).alias('a'))
    p = p.with_columns(pl.col('a').alias('b')).explode('a').explode('b').drop_nulls().filter(pl.col('a') < pl.col('b'))
    add(p, 'pair', pl.col('a') + ' ' + pl.col('b'))
    cat = base.select('id', pl.col('ct').list.join('').alias('cc'), 'dom')
    add(cat.filter(pl.col('cc').str.len_chars() >= 5), 'cat', pl.col('cc').str.slice(0, 10))
    add(cat.filter(pl.col('dom').str.len_chars() >= 5), 'cat', pl.col('dom').str.slice(0, 10))
    add(base.filter(pl.col('ct').list.len() > 0), 'full', pl.col('ct').list.sort().list.join(' '))
    nw = t.join(adw.select('id', 'w'), on='id')
    add(nw, 'nw', pl.col('t') + '|' + pl.col('w'))
    a4 = adw.group_by('id', maintain_order=True).head(5).select('id', pl.col('w').alias('a'))
    ww = a4.join(a4.rename({'a': 'b'}), on='id').filter(pl.col('a') < pl.col('b'))
    add(ww, 'ww', pl.col('a') + '|' + pl.col('b'))
    h = base.select('id', 'hl').explode('hl').drop_nulls().join(adw.select('id', 'w'), on='id')
    add(h, 'hw', pl.col('hl') + '|' + pl.col('w'))
    # full name x any address word (common names disambiguated by city)
    fw = base.filter(pl.col('ct').list.len() > 0).select('id', pl.col('ct').list.sort().list.join(' ').alias('f')) \
        .join(adall.select('id', 'w'), on='id')
    add(fw, 'fw', pl.col('f') + '|' + pl.col('w'))
    add(base.select('id', 'codes').explode('codes').drop_nulls(), 'code', pl.col('codes'))
    # 6-char substrings: concatenated names / domains on side 1, prefixes of long tokens on side 0
    cc1 = base.filter(pl.col('side') == 1).select('id', pl.when(pl.col('dom') != '').then(pl.col('dom')).otherwise(
        pl.when(pl.col('ct').list.len() == 1).then(pl.col('ct').list.first()).otherwise(pl.lit(''))).alias('c'))
    cc1 = cc1.filter(pl.col('c').str.len_chars() >= 9).with_columns(
        o=pl.int_ranges(0, pl.col('c').str.len_chars() - 5)).explode('o')
    add(cc1, 'sub', pl.col('c').str.slice(pl.col('o'), 6))
    s0 = t.join(sd.filter(pl.col('side') == 0), on='id').filter(pl.col('t').str.len_chars() >= 6)
    add(s0, 'sub', pl.col('t').str.slice(0, 6))
    k = pl.concat(out).unique(['id', 'key']).join(sd, on='id')
    return k


def gen_country(split, ctry):
    """Key-blocking candidate pairs of one country."""
    return gen_pairs(load(split, ctry), ctry)


def gen_pairs(rec, ctry=''):
    """rec: (id, side, country, core, dom, ad, hn, state) -> candidate pairs (id1, id2, ks, kbits, nk)."""
    t0 = time.time()
    k = keys(rec)
    sz = k.group_by('key').agg((pl.col('side') == 0).sum().alias('n1'), (pl.col('side') == 1).sum().alias('n2'),
                               pl.col('fam').first())
    caps = pl.DataFrame({'fam': list(FAM), 'c1': [v[1] for v in FAM.values()], 'c2': [v[2] for v in FAM.values()],
                         'bit': [v[0] for v in FAM.values()]})
    sz = (sz.join(caps, on='fam').filter((pl.col('n1') >= 1) & (pl.col('n2') >= 1) & (pl.col('n1') <= pl.col('c1'))
                                         & (pl.col('n2') <= pl.col('c2')))
          .with_columns(w=1.0 / (pl.col('n1') * pl.col('n2')).cast(pl.Float32).sqrt()))
    k = k.join(sz.select('key', 'w', 'bit'), on='key')
    a = k.filter(pl.col('side') == 0).select(pl.col('id').alias('id1'), 'key', 'w', 'bit')
    b = k.filter(pl.col('side') == 1).select(pl.col('id').alias('id2'), 'key')
    pr = (a.join(b, on='key').group_by('id1', 'id2')
          .agg(pl.col('w').sum().alias('ks'), pl.col('bit').unique().sum().alias('kbits'), pl.len().alias('nk')))
    print(ctry, 'recs', rec.height, 'keys', k.height, 'pairs', pr.height, f'{time.time() - t0:.0f}s', flush=True)
    return pr.with_columns(kbits=pl.col('kbits').cast(pl.Int16), nk=pl.col('nk').cast(pl.Int16))


def cap(pr, k1=K1, k2=K2):
    """Keep a pair if it is among the k1 best of its S1 or among the k2 best S1 of its S2/S3 record."""
    r1 = pl.col('ks').rank('ordinal', descending=True)
    pr = pr.with_columns(r1=r1.over('id1').cast(pl.Int32), r2=r1.over('id2').cast(pl.Int32))
    return pr.filter((pl.col('r1') <= k1) | (pl.col('r2') <= k2))


def gen(split):
    """Blocking for every country of a split: uncapped pairs per country and the capped (K1/K2) set."""
    ctrs = pl.read_parquet(f'{W}/n_{split}_s1.parquet', columns=['country'])['country'].unique().sort().to_list()
    allp = []
    for c in ctrs:
        pr = gen_country(split, c)
        pr.write_parquet(f'{W}/cand0_{split}_{c}.parquet')
        allp.append(cap(pr))
    cand = pl.concat(allp)
    cand.write_parquet(f'{W}/cand_{split}.parquet')
    print('capped candidates', cand.height, flush=True)


def report(split, name='cand_train', grid=False):
    """Recall, ceiling F0.5 and candidates per S1 of a candidate set, overall, per country, per key family."""
    s1 = pl.read_parquet(f'{W}/n_{split}_s1.parquet', columns=['id', 'country']).with_columns(s1=enc('id'))
    gt = pl.read_parquet(gt_path(split)).select(enc('s1').alias('s1'), enc('id2').alias('id2'))

    def rep(pr, tag):
        """Print recall, ceiling F0.5 and candidate counts of the pair set pr."""
        hit = gt.join(pr.select(pl.col('id1').alias('s1'), 'id2'), on=['s1', 'id2'], how='semi')
        per = (s1.join(gt.group_by('s1').len('nt'), on='s1', how='left')
               .join(hit.group_by('s1').len('nh'), on='s1', how='left')
               .join(pr.group_by('id1').len('nc').rename({'id1': 's1'}), on='s1', how='left').fill_null(0))
        for ctry, g in [('ALL', per)] + [(c, per.filter(pl.col('country') == c)) for c in per['country'].unique().sort()]:
            f, _ = macro_fbeta_arr(g['nt'], g['nh'], g['nh'])
            print(f'{tag} {ctry:6s} recall={g["nh"].sum() / max(g["nt"].sum(), 1):.4f} ceilF05={f:.4f} '
                  f'cand/S1={g["nc"].mean():.1f} p99={g["nc"].quantile(0.99):.0f}', flush=True)

    pr = pl.read_parquet(f'{W}/{name}.parquet', columns=['id1', 'id2', 'kbits', 'ks'])
    rep(pr, name)
    h = gt.join(pr.select(pl.col('id1').alias('s1'), 'id2', 'kbits'), on=['s1', 'id2'])
    for n, v in FAM.items():
        print(f'  fam {n:5s} hits={h.filter((pl.col("kbits") & v[0]) > 0).height / gt.height:.4f} '
              f'only={h.filter(pl.col("kbits") == v[0]).height / gt.height:.4f}')
    if grid:
        full = pl.concat([pl.read_parquet(f'{W}/cand0_{split}_{c}.parquet') for c in s1['country'].unique()])
        rep(full, 'uncapped')
        for k1, k2 in ((20, 3), (30, 4), (40, 6), (60, 8), (100, 10)):
            rep(cap(full, k1, k2), f'K1={k1} K2={k2}')


if __name__ == '__main__':
    cmd, split = sys.argv[1], sys.argv[2]
    if cmd == 'gen':
        gen(split)
        if split != 'test':
            report(split, f'cand_{split}', grid=True)
    else:
        report(split, sys.argv[3] if len(sys.argv) > 3 else 'cand_train', grid='grid' in sys.argv)
