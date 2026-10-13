"""Pair features for (S1, S2/S3) candidates. Country-agnostic (country never used as a feature).

python feats.py split=tsim cand=cand_tsim_k100 [parts=24]  -> W/f_{cand}/p_XX.parquet (pairs + source + features)
"""
import os
import sys
import math
import time
from multiprocessing import Pool

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

W = os.environ.get('ER_WORK', 'work')
FCOLS = ['nm', 'core', 'legal', 'alt', 'dom', 'ad', 'hn', 'pc', 'state', 'nind', 'nunk']

_IDF = {}


def idf_table(split):
    """token -> idf over all records of the split (names + address words)."""
    fr = []
    for k in (1, 2, 3):
        d = pl.read_parquet(f'{W}/n_{split}_s{k}.parquet', columns=['core', 'ad'])
        fr.append(d.select((pl.col('core') + ' ' + pl.col('ad').str.replace_all(r'[^a-z0-9 ]', ' ')).alias('t')))
    t = pl.concat(fr)
    n = t.height
    t = t.with_columns(pl.col('t').str.split(' ').list.unique()).explode('t')
    df = t.filter(pl.col('t') != '').group_by('t').len()
    return df, n


def _init(idf):
    """Pool worker initializer: share the IDF table."""
    global _IDF
    _IDF = idf


def widf(a, b):
    """idf-weighted jaccard of token sets."""
    if not a or not b:
        return -1.0
    inter = a & b
    uni = a | b
    num = sum(_IDF.get(t, 12.0) for t in inter)
    den = sum(_IDF.get(t, 12.0) for t in uni)
    return num / den if den else 0.0


def soft_cov(a, b, th=0.88):
    """fraction of tokens of a with a JW>=th match in b."""
    if not a or not b:
        return -1.0
    c = 0
    for x in a:
        if x in b:
            c += 1
            continue
        for y in b:
            if JaroWinkler.similarity(x, y) >= th:
                c += 1
                break
    return c / len(a)


def acro(t):
    """Acronym of a token list."""
    return ''.join(w[0] for w in t if w)


def pair_row(r):
    """All string-similarity features of one (S1, S2/S3) pair, in the order of NAMES."""
    (nm1, co1, lg1, al1, dm1, ad1, hn1, pc1, st1, _, _,
     nm2, co2, lg2, al2, dm2, ad2, hn2, pc2, st2, ni2, nu2) = r
    f = []
    ct1, ct2 = co1.split(), co2.split()
    s1, s2 = set(ct1), set(ct2)
    cc1, cc2 = ''.join(ct1), ''.join(ct2)
    # ---- name
    f += [fuzz.ratio(nm1, nm2), fuzz.token_set_ratio(nm1, nm2), fuzz.token_sort_ratio(nm1, nm2),
          fuzz.partial_ratio(nm1, nm2) if nm1 and nm2 else -1,
          fuzz.ratio(co1, co2), fuzz.token_set_ratio(co1, co2), fuzz.token_sort_ratio(co1, co2),
          JaroWinkler.similarity(co1, co2), fuzz.ratio(cc1, cc2),
          fuzz.partial_ratio(cc1, cc2) if cc1 and cc2 else -1,
          len(s1 & s2) / len(s1 | s2) if s1 and s2 else -1,
          widf(s1, s2), soft_cov(ct1, ct2), soft_cov(ct2, ct1),
          float(bool(ct1) and bool(ct2) and ct1[0] == ct2[0]),
          float(acro(ct1) == cc2) if len(ct1) >= 2 else -1]
    # domain / alt names
    f += [fuzz.ratio(cc1, dm2) if dm2 else -1, fuzz.partial_ratio(dm2, cc1) if dm2 and cc1 else -1,
          fuzz.token_set_ratio(co1, al2) if al2 else -1, fuzz.token_set_ratio(al1, co2) if al1 else -1]
    # legal form
    l1, l2 = set(lg1.split()), set(lg2.split())
    f += [float(l1 == l2) if l1 and l2 else -1, float(bool(l1) and bool(l2) and not (l1 & l2)),
          float(bool(l1)), float(bool(l2))]
    f += [len(ct1), len(ct2), len(nm1), len(nm2), ni2, nu2]
    # ---- address
    e1, e2 = not ad1, not ad2
    if e1 or e2:
        f += [-1] * 16
    else:
        w1 = set(ad1.replace(',', ' ').split())
        w2 = set(ad2.replace(',', ' ').split())
        a1 = {t for t in w1 if not any(ch.isdigit() for ch in t)}
        a2 = {t for t in w2 if not any(ch.isdigit() for ch in t)}
        h1, h2 = set(hn1.split()), set(hn2.split())
        c1 = [c.strip() for c in ad1.split(',')]
        c2 = [c.strip() for c in ad2.split(',')]
        cm = np.mean([max(fuzz.ratio(x, y) for y in c2) for x in c1])
        cm2 = np.mean([max(fuzz.ratio(x, y) for y in c1) for x in c2])
        f += [fuzz.token_set_ratio(ad1, ad2), fuzz.token_sort_ratio(ad1, ad2), fuzz.ratio(ad1, ad2),
              len(a1 & a2) / len(a1 | a2) if a1 and a2 else -1, widf(a1, a2),
              soft_cov(list(a1), list(a2)), soft_cov(list(a2), list(a1)), cm, cm2,
              len(h1 & h2) if h1 and h2 else -1,
              float(bool(h1) and bool(h2) and not (h1 & h2)),
              float(h1 <= h2) if h1 and h2 else -1,
              float(hn1.split()[0] == hn2.split()[0]) if h1 and h2 else -1,
              float(bool(pc1) and bool(pc2) and pc1 == pc2) if pc1 and pc2 else -1,
              float(st1 == st2) if st1 and st2 else -1,
              float(len(h2) == 0)]
    f += [float(e1), float(e2)]
    # cross: name tokens found inside the other address (landmark / name-as-address noise)
    f += [len(s2 & set(ad1.split())) / max(len(s2), 1)]
    return f


NAMES = (['nm_r', 'nm_ts', 'nm_tso', 'nm_pr', 'co_r', 'co_ts', 'co_tso', 'co_jw', 'cc_r', 'cc_pr', 'co_jac',
          'co_widf', 'co_cov12', 'co_cov21', 'co_first', 'acro',
          'dom_r', 'dom_pr', 'alt2_ts', 'alt1_ts',
          'lg_eq', 'lg_conf', 'lg1', 'lg2', 'nt1', 'nt2', 'nl1', 'nl2', 'nind2', 'nunk2',
          'ad_ts', 'ad_tso', 'ad_r', 'ad_jac', 'ad_widf', 'ad_cov12', 'ad_cov21', 'ad_cm12', 'ad_cm21',
          'hn_inter', 'hn_conf', 'hn_sub', 'hn_first', 'pc_eq', 'st_eq', 'hn2_empty',
          'ad1_empty', 'ad2_empty', 'nm2_in_ad1'])


def _chunk(rows):
    """Features of a chunk of pairs (runs in a worker process)."""
    return np.asarray([pair_row(r) for r in rows], dtype=np.float32)


def load_fields(split):
    """Normalized fields of the S1 and of the S2/S3 records of a split, keyed by encoded id."""
    sys.path.insert(0, os.path.dirname(__file__))
    from block import enc
    s1 = pl.read_parquet(f'{W}/n_{split}_s1.parquet', columns=['id'] + FCOLS).with_columns(id1=enc('id')).drop('id')
    s23 = pl.concat([pl.read_parquet(f'{W}/n_{split}_s{k}.parquet', columns=['id'] + FCOLS) for k in (2, 3)])
    s23 = s23.with_columns(id2=enc('id')).drop('id').rename({c: c + '_2' for c in FCOLS})
    return s1, s23


def compute(pairs, split, out, nproc=32, nparts=24):
    """pairs: DataFrame id1,id2 (int64) + extra cols. Writes out_{i}.parquet parts (pairs + src + features)."""
    t0 = time.time()
    s1, s23 = load_fields(split)
    idf, n = idf_table(split)
    idfd = dict(zip(idf['t'].to_list(), (np.log(n / idf['len'].to_numpy())).tolist()))
    del idf
    cols = FCOLS + [c + '_2' for c in FCOLS]
    pairs = pairs.with_columns(part=(pl.col('id1').hash(seed=3) % nparts).cast(pl.Int16))
    print('idf', len(idfd), f'{time.time() - t0:.0f}s', flush=True)
    step = 4000
    with Pool(nproc, initializer=_init, initargs=(idfd,)) as pool:
        for i in range(nparts):
            pp = pairs.filter(pl.col('part') == i).drop('part')
            d = pp.join(s1, on='id1', how='left').join(s23, on='id2', how='left')
            d = d.with_columns([pl.col(c).fill_null('') for c in cols if d[c].dtype == pl.String]
                               + [pl.col(c).fill_null(0) for c in cols if d[c].dtype != pl.String])
            rows = list(d.select(cols).iter_rows())
            del d
            X = np.vstack(pool.map(_chunk, [rows[j:j + step] for j in range(0, len(rows), step)], chunksize=2))
            del rows
            fd = pl.DataFrame(X, schema=NAMES)
            src = pp.select((pl.col('id2') // 10_000_000_000).cast(pl.Int8).alias('src'))
            pl.concat([pp, src, fd], how='horizontal').write_parquet(f'{out}_{i:02d}.parquet')
            print('part', i, pp.height, f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    kv = dict(a.split('=', 1) for a in sys.argv[1:])
    split, cand = kv.get('split', 'train'), kv['cand']
    pr = pl.read_parquet(f'{W}/{cand}.parquet')
    if 'n' in kv:
        pr = pr.head(int(kv['n']))
    os.makedirs(f'{W}/f_{cand}', exist_ok=True)
    compute(pr, split, f'{W}/f_{cand}/p', int(os.environ.get('SLURM_CPUS_PER_TASK', 32)),
            int(kv.get('parts', 24)))
