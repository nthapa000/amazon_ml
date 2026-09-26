"""Name-conflict features (C) for the cascade pairs, motivated by the French borderline errors on test:
- word swap: same address, one distinctive name word swapped (ZE Ecole SARL vs ZE Club SARL)
- street conflict: same name, different street (44 Rue de Normandie vs 16 Rue de Roeun)
- location in the name: exact-name records with an empty address, names that embed the S1 city (Pessac College SARL)
- sibling conflict: the candidate's name disagrees with the S1's confident matches
All features are plain string comparisons of the pair's own records, identical for every country; word frequencies
come from the split's own S1 names (like the other split-statistics features).

python feats3.py split=tsim pred=oof_b1ka thr=0.001 out=f3_tsim [frac=0.2 report=1 fin=fin2b]
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import polars as pl
from rapidfuzz import fuzz

sys.path.insert(0, os.path.dirname(__file__))
from block import enc, gt_path  # noqa: E402

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')
STOP = {'the', 'and', 'of', 'a', 'an', 'de', 'du', 'des', 'la', 'le', 'les', 'et', 'l', 'd', 'ms', 'm/s', 'no', 'n'}
TYPES = {'street', 'road', 'avenue', 'boulevard', 'lane', 'drive', 'court', 'place', 'plaza', 'square', 'highway',
         'parkway', 'route', 'circle', 'way', 'trail', 'terrace', 'alley', 'crescent', 'expressway', 'freeway', 'pike',
         'loop', 'run', 'marg', 'nagar', 'sector', 'colony', 'main', 'cross', 'layout', 'rue', 'allee', 'chemin',
         'impasse', 'quai', 'cours', 'faubourg', 'passage', 'promenade', 'sente', 'montee', 'residence', 'lotissement',
         'zone', 'bis', 'ter', 'north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest',
         'floor', 'building', 'apartment', 'suite', 'unit', 'number', 'near', 'opposite', 'plot', 'shop'}
NF = ['u1', 'u2', 'swap', 'fq1', 'fq2', 'st_r', 'st_conf', 'nm_eq', 'nm_eq_stc', 'city_n12', 'e2_nmeq', 'e2_city',
      'sib_n', 'sib_u2', 'sib_swap', 'nmc1']
RF = {}


def rec(split, k):
    """per record: name tokens, street-name string, city tokens, empty-address flag."""
    d = pl.read_parquet(f'{W}/n_{split}_s{k}.parquet', columns=['id', 'country', 'nm', 'core', 'ad', 'state'])
    rows = []
    for i, c, core, ad, st in zip(d['id'], d['country'], d['core'], d['ad'], d['state']):
        t = sorted({x for x in core.split() if x and x not in STOP})
        comps = [x.strip() for x in ad.split(' , ') if x.strip()] if ad else []
        sc = next((x for x in comps if any(ch.isdigit() for ch in x)), None)
        if sc is None:
            sc = next((x for x in comps if set(x.split()) & TYPES), None)
        sn = ' '.join(sorted(w for w in sc.split() if w.isalpha() and w not in TYPES and w not in STOP
                             and len(w) > 1)) if sc else ''
        cty = sorted({w for x in comps if x != sc and x != st for w in x.split()
                      if w.isalpha() and len(w) > 2 and w not in STOP and w not in TYPES})
        rows.append((i, c, t, sn, cty, not ad))
    return pl.DataFrame(rows, schema=['id', 'country', 'tok', 'sn', 'cty', 'emp'], orient='row').with_columns(id=enc('id'))


def resid(a, b):
    """unmatched name words on each side, tolerant of typos (ratio >= 80, len >= 4) and truncation (prefix >= 3)."""
    r1, r2 = [x for x in a if x not in b], [x for x in b if x not in a]
    for x in list(r1):
        for y in r2:
            if (len(x) >= 4 and len(y) >= 4 and fuzz.ratio(x, y) >= 80) or \
                    (min(len(x), len(y)) >= 3 and (x.startswith(y) or y.startswith(x))):
                r1.remove(x)
                r2.remove(y)
                break
    return r1, r2


def work(rows):
    out = []
    for c, t1, t2, s1, s2, cy1, e2, sibs in rows:
        rf = RF.get(c, {})
        r1, r2 = resid(t1, t2)
        u1, u2 = len(r1), len(r2)
        fq1 = max((rf.get(x, -7.0) for x in r1), default=-9.0)
        fq2 = max((rf.get(x, -7.0) for x in r2), default=-9.0)
        if s1 and s2:
            sr = fuzz.token_set_ratio(s1, s2) / 100
            sc = float(sr < 0.6)
        else:
            sr, sc = -1.0, -1.0
        eq = float(u1 == 0 and u2 == 0 and len(t1) > 0)
        cn = float(any(x in t1 for x in cy1) and any(x in t2 for x in cy1))
        su2, ssw = 9.0, 1.0
        for st in sibs:
            a, b = resid(st, t2)
            su2 = min(su2, len(b))
            ssw = min(ssw, float(len(a) > 0 and len(b) > 0))
        out.append((u1, u2, float(u1 > 0 and u2 > 0), fq1, fq2, sr, sc, eq, float(eq == 1.0 and sc == 1.0), cn,
                    float(e2 and eq == 1.0), float(e2 and cn == 1.0), float(len(sibs)),
                    su2 if sibs else -1.0, ssw if sibs else -1.0))
    return out


def main(kv):
    t0 = time.time()
    sp = kv['split']
    thr = float(kv.get('thr', 0.001))
    s1 = rec(sp, 1)
    s23 = pl.concat([rec(sp, k) for k in (2, 3)])
    print('records', s1.height, s23.height, f'{time.time() - t0:.0f}s', flush=True)
    # word frequency of S1 name words per country: log10((df + 1) / N)
    wf = s1.select('country', 'tok').explode('tok').drop_nulls().group_by('country', 'tok').len('df')
    nc = s1.group_by('country').len('N')
    wf = wf.join(nc, on='country').with_columns(rf=((pl.col('df') + 1) / pl.col('N')).log10())
    for c in wf['country'].unique():
        g = wf.filter(pl.col('country') == c)
        RF[c] = dict(zip(g['tok'], g['rf']))
    nm = pl.read_parquet(f'{W}/n_{sp}_s1.parquet', columns=['id', 'nm']).with_columns(id=enc('id'))
    nmc = nm.join(nm.group_by('nm').len('c'), on='nm').select(pl.col('id').alias('id1'),
                                                               nmc1=pl.col('c').cast(pl.Float32).log1p())
    pr = pl.scan_parquet(f'{W}/{kv["pred"]}.parquet').select('id1', 'id2', 'p').collect()
    pairs = pr.filter(pl.col('p') >= thr).select('id1', 'id2')
    if 'frac' in kv:
        pairs = pairs.filter(pl.col('id1').hash(seed=17) % 1000 < int(float(kv['frac']) * 1000))
    sib = (pr.filter(pl.col('p') >= 0.9).join(s23.select(pl.col('id').alias('id2'), pl.col('tok').alias('stok')), on='id2')
           .select('id1', 'id2', 'stok'))
    del pr
    print('pairs', pairs.height, 'sibling rows', sib.height, f'{time.time() - t0:.0f}s', flush=True)
    a = s1.select(pl.col('id').alias('id1'), 'country', pl.col('tok').alias('t1'), pl.col('sn').alias('s1'),
                  pl.col('cty').alias('c1'))
    b = s23.select(pl.col('id').alias('id2'), pl.col('tok').alias('t2'), pl.col('sn').alias('s2'), pl.col('emp').alias('e2'))
    res = []
    nproc = int(os.environ.get('SLURM_CPUS_PER_TASK', 16))
    with Pool(nproc) as pool:
        for i in range(0, pairs.height, 1_000_000):
            d = pairs.slice(i, 1_000_000).join(a, on='id1', how='left').join(b, on='id2', how='left')
            sg = (d.select('id1', 'id2').join(sib.rename({'id2': 'sid2'}), on='id1')
                  .filter(pl.col('sid2') != pl.col('id2')).group_by('id1', 'id2').agg(pl.col('stok').head(5)))
            d = d.join(sg, on=['id1', 'id2'], how='left')
            rows = list(zip(d['country'], d['t1'].to_list(), d['t2'].to_list(), d['s1'], d['s2'], d['c1'].to_list(),
                            d['e2'], [x or [] for x in d['stok'].to_list()]))
            rows = [(c, t1 or [], t2 or [], s1_ or '', s2_ or '', c1 or [], bool(e2), sb)
                    for c, t1, t2, s1_, s2_, c1, e2, sb in rows]
            ch = [rows[j:j + 20000] for j in range(0, len(rows), 20000)]
            f = [x for part in pool.map(work, ch) for x in part]
            res.append(d.select('id1', 'id2').hstack(pl.DataFrame(f, schema=NF[:-1], orient='row').cast(pl.Float32)))
            print(f'  {i + len(rows)} pairs {time.time() - t0:.0f}s', flush=True)
    f3 = pl.concat(res).join(nmc, on='id1', how='left')
    f3.write_parquet(f'{W}/{kv["out"]}.parquet')
    print('wrote', kv['out'], f3.shape, f'{time.time() - t0:.0f}s', flush=True)
    if kv.get('report') == '1':
        report(f3, sp, kv.get('fin', 'fin2b'))


def report(f3, sp, fin):
    """discriminativeness on labelled pairs: prevalence and match rate per feature value, overall and in the
    uncertain band of the final model (0.1 < p < 0.9)."""
    gt = pl.read_parquet(gt_path(sp)).select(enc('s1').alias('id1'), enc('id2').alias('id2'), y=pl.lit(1, pl.Int8))
    o = pl.read_parquet(f'{W}/oof_{fin}.parquet').select('id1', 'id2', pl.col('p').alias('pf'))
    d = f3.join(o, on=['id1', 'id2'], how='left').join(gt, on=['id1', 'id2'], how='left').with_columns(pl.col('y').fill_null(0))
    band = d.filter((pl.col('pf') > 0.1) & (pl.col('pf') < 0.9))
    print(f'labelled pairs {d.height}, positives {int(d["y"].sum())}; final-model band pairs {band.height}, '
          f'positives {int(band["y"].sum())}', flush=True)
    for col, cut in (('swap', 0.5), ('st_conf', 0.5), ('nm_eq_stc', 0.5), ('e2_nmeq', 0.5), ('e2_city', 0.5),
                     ('city_n12', 0.5), ('sib_swap', 0.5), ('u2', 0.5), ('fq2', -2.0)):
        for nm_, x in (('all', d), ('band', band)):
            g = x.with_columns(v=(pl.col(col) > cut)).group_by('v').agg(n=pl.len(), rate=pl.col('y').mean()).sort('v')
            txt = '  '.join(f'{col}>{cut}={v}: n={n} match={r:.3f}' for v, n, r in g.iter_rows())
            print(f'{nm_:4s} {txt}', flush=True)


if __name__ == '__main__':
    main(dict(a.split('=', 1) for a in sys.argv[1:]))
