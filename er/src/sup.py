"""Collective ER: sibling-support features for cascade pairs.

Siblings of an S1 = its confident matches from stage-1 scores (OOF on tsim, fold-averaged on test):
one-to-one (margin 0.2) and p >= sib_thr. For each cascade pair (s1, x), x is compared with the other siblings
of s1 (never with itself): normalized and raw name/address similarity, house-number agreement.
Records of one business often share raw text (same native-script name, same source formatting), which is
evidence that S1-vs-x similarity alone can miss; a distractor usually disagrees with the siblings.

python sup.py split=tsim pred=oof_w1 thr=0.001 sib_thr=0.9   -> W/sup_{split}.parquet
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

sys.path.insert(0, os.path.dirname(__file__))
from block import enc  # noqa: E402
from decide import one2one  # noqa: E402

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')
COLS = ['nm', 'core', 'ad', 'hn', 'name', 'addr']
NAMES = ['s_nts', 's_njw', 's_ats', 's_hn', 's_rn', 's_rneq', 's_ra', 's_raeq']


def row(r):
    nm1, co1, ad1, hn1, rn1, ra1, nm2, co2, ad2, hn2, rn2, ra2 = r
    e = not ad1 or not ad2
    h1, h2 = hn1.split()[:1], hn2.split()[:1]
    return (fuzz.token_set_ratio(nm1, nm2), JaroWinkler.similarity(co1, co2),
            -1.0 if e else fuzz.token_set_ratio(ad1, ad2),
            -1.0 if not h1 or not h2 else float(h1 == h2),
            fuzz.ratio(rn1, rn2), float(rn1 == rn2 and rn1 != ''),
            -1.0 if not ra1 or not ra2 else fuzz.ratio(ra1, ra2), float(ra1 == ra2 and ra1 != ''))


def chunk(rows):
    return np.asarray([row(r) for r in rows], dtype=np.float32)


def main(kv):
    t0 = time.time()
    split, thr, st = kv['split'], float(kv.get('thr', 1e-3)), float(kv.get('sib_thr', 0.9))
    pr = pl.read_parquet(f'{W}/{kv["pred"]}.parquet', columns=['id1', 'id2', 'p'])
    conf = one2one(pr, 0.2).filter(pl.col('p') >= st).select('id1', pl.col('id2').alias('sib'))
    pairs = pr.filter(pl.col('p') >= thr).select('id1', 'id2')
    del pr
    trip = pairs.join(conf, on='id1').filter(pl.col('id2') != pl.col('sib'))
    print('pairs', pairs.height, 'siblings', conf.height, 'comparisons', trip.height, flush=True)
    rec = pl.concat([pl.read_parquet(f'{W}/n_{split}_s{k}.parquet', columns=['id'] + COLS) for k in (2, 3)])
    rec = rec.with_columns(id=enc('id')).with_columns([pl.col(c).fill_null('') for c in COLS])
    need = pl.concat([trip.select(pl.col('id2').alias('id')), trip.select(pl.col('sib').alias('id'))]).unique()
    rec = rec.join(need, on='id', how='semi')
    t = (trip.join(rec.rename({c: c + '_x' for c in COLS}).rename({'id': 'id2'}), on='id2', how='left')
         .join(rec.rename({c: c + '_s' for c in COLS}).rename({'id': 'sib'}), on='sib', how='left'))
    cols = [c + '_x' for c in COLS] + [c + '_s' for c in COLS]
    t = t.with_columns([pl.col(c).fill_null('') for c in cols])
    step, big, parts = 20000, 2_000_000, []
    with Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', 32))) as pool:
        for j in range(0, t.height, big):             # slices keep the python row objects small
            rows = list(t.slice(j, big).select(cols).iter_rows())
            parts.append(np.vstack(pool.map(chunk, [rows[i:i + step] for i in range(0, len(rows), step)],
                                            chunksize=2)))
            print('  compared', j + len(rows), f'{time.time() - t0:.0f}s', flush=True)
    X = np.vstack(parts)
    f = t.select('id1', 'id2', 'sib').with_columns([pl.Series(n, X[:, i]) for i, n in enumerate(NAMES)])
    agg = f.group_by('id1', 'id2').agg(
        pl.len().cast(pl.Int16).alias('s_n'),
        ((pl.col('sib') // 10_000_000_000) == (pl.col('id2') // 10_000_000_000)).sum().cast(pl.Int16).alias('s_nsame'),
        pl.col('s_nts').max().alias('s_nts_max'), pl.col('s_nts').mean().alias('s_nts_mean'),
        pl.col('s_njw').max().alias('s_njw_max'),
        pl.col('s_ats').max().alias('s_ats_max'), pl.col('s_ats').mean().alias('s_ats_mean'),
        pl.col('s_hn').max().alias('s_hn_any'),
        pl.col('s_rn').max().alias('s_rn_max'), pl.col('s_rneq').max().alias('s_rneq_any'),
        pl.col('s_ra').max().alias('s_ra_max'), pl.col('s_raeq').max().alias('s_raeq_any'),
        pl.min_horizontal('s_nts', pl.col('s_ats').clip(0)).max().alias('s_both_max'))
    out = pairs.join(agg, on=['id1', 'id2'], how='left').with_columns(pl.col('s_n').fill_null(0), pl.col('s_nsame').fill_null(0))
    out.write_parquet(f'{W}/{kv.get("out", "sup_" + split)}.parquet')
    print('written', out.height, f'{time.time() - t0:.0f}s', flush=True)
    print(out.drop('id1', 'id2').describe())


if __name__ == '__main__':
    main(dict(a.split('=', 1) for a in sys.argv[1:]))
