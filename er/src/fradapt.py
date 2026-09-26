"""Self-training set for the raw-text cross-encoder on countries unseen in training (L1, 'noisy student').
Built only from the provided records; organizers' Q&A (25-26 Sep): self-training on our own test predictions and
synthetic pairs generated from the provided records are allowed.
For S1 entities whose country never occurs in training (decide.unseen), from the teacher's test scores (pte_{tag}):
- pos:   teacher-confident matches: one-to-one with margin 0.3 and p >= 0.97
- neg:   teacher-confident non-matches that look alike: p <= 0.02 and max(name, address) token_set_ratio >= 80
- s1neg: two distinct S1 records (Source 1 is deduplicated -> different businesses: exact labels) that share the
         normalized address or the core name; one side is noised so that formatting cannot give the label away
- syn:   an S1 record vs a noised copy of itself (exact positive)
The pos / neg cutoffs are also applied to tsim (labelled) to report pseudo-label precision.
python fradapt.py tag=fin2 out=fra_b1ka [npos=200000 nneg=200000 ns1=150000 nsyn=100000]
-> W/{out}.parquet (t1, t2, y, kind), texts in the ce.py raw format
"""
import os
import re
import sys
import json
import random

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process, utils
from unidecode import unidecode

sys.path.insert(0, os.path.dirname(__file__))
from block import enc, gt_path  # noqa: E402
from decide import one2one, unseen  # noqa: E402
import norm  # noqa: E402

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')
RND = random.Random(7)
ABR = [(r'\brue\b', 'R.'), (r'\bavenue\b', 'Av.'), (r'\bboulevard\b', 'Bd'), (r'\ball[ée]e\b', 'All.'),
       (r'\bplace\b', 'Pl.'), (r'\bimpasse\b', 'Imp.'), (r'\broute\b', 'Rte'), (r'\bchemin\b', 'Ch.'),
       (r'\bstreet\b', 'St'), (r'\broad\b', 'Rd')]
LEGAL = [('SARL', 'S.A.R.L.'), ('SASU', 'S.A.S.U.'), ('SAS', 'S.A.S.'), ('SCI', 'S.C.I.'), ('SA', 'S.A.')]
REG = [(re.compile(r.replace(' ', r'[- ]'), re.I), ds.split('|')) for r, ds in norm._FR_DEP.items()]


def fmt(n, a):
    return (f'name: {n} | address: {a if a else "none"}')[:200]


def noise(n, a):
    """formatting noise of the kind seen between sources (hand-written, no labels involved)"""
    if a:
        if RND.random() < 0.5:
            for rx, ds in REG:
                if rx.search(a):
                    a = rx.sub(RND.choice(ds).title().replace(' ', '-'), a)
                    break
        if RND.random() < 0.5:
            for rx, s in ABR:
                a = re.sub(rx, s, a, flags=re.I)
        parts = [x.strip() for x in a.split(',') if x.strip()]
        if len(parts) > 2 and RND.random() < 0.25:
            parts = parts[:-1]
        if len(parts) > 1 and RND.random() < 0.25:
            parts = parts[1:] + parts[:1]
        a = ', '.join(parts)
        if RND.random() < 0.2:
            a = re.sub(r'^(\d+)', r'N°\1', a)
        if RND.random() < 0.3:
            a = a.upper()
        if RND.random() < 0.3:
            a = unidecode(a)
    for s, d in LEGAL:
        if re.search(rf'\b{s}\b', n):
            r = RND.random()
            n = re.sub(rf'\b{s}\b', d if r < 0.3 else ('' if r < 0.5 else s), n).strip()
            break
    if RND.random() < 0.2 and len(n) > 6:
        i = RND.randrange(1, len(n) - 1)
        n = n[:i] + n[i + 1:]
    if RND.random() < 0.3:
        n = unidecode(n)
    if RND.random() < 0.2:
        n = n.upper()
    return n, a


def recs(sp, ks):
    return pl.concat([pl.read_parquet(f'{W}/n_{sp}_s{k}.parquet', columns=['id', 'name', 'addr']) for k in ks]).select(
        enc('id').alias('id'), pl.col('name').fill_null(''), pl.col('addr').fill_null(''))


def attach(p, r1, r2):
    return (p.join(r1.rename({'id': 'id1', 'name': 'n1', 'addr': 'a1'}), on='id1')
            .join(r2.rename({'id': 'id2', 'name': 'n2', 'addr': 'a2'}), on='id2'))


def lookalike(d):
    sn = process.cpdist(d['n1'].to_list(), d['n2'].to_list(), scorer=fuzz.token_set_ratio,
                        processor=utils.default_process, workers=4)
    sa = process.cpdist(d['a1'].to_list(), d['a2'].to_list(), scorer=fuzz.token_set_ratio,
                        processor=utils.default_process, workers=4)
    return d.with_columns(sim=pl.Series(np.maximum(sn, sa)))


def pools(p):
    pos = one2one(p, 0.3).filter(pl.col('p') >= 0.97).select('id1', 'id2')
    neg = p.filter(pl.col('p') <= 0.02).select('id1', 'id2')
    return pos, neg


def adjacent(d, key):
    """consecutive pairs of S1 records sharing `key` (bounded: no quadratic blow-up in big groups)"""
    d = d.filter(pl.col(key) != '').sort(key, 'id').with_columns(r=pl.col('id').cum_count().over(key))
    b = d.select(key, (pl.col('r') - 1).alias('r'), pl.col('id').alias('idb'))
    return d.select(key, 'r', 'id').join(b, on=[key, 'r']).select(pl.col('id').alias('ida'), 'idb')


def main(kv):
    tag, out = kv.get('tag', 'fin2'), kv['out']
    cfg = json.load(open(f'{W}/dec_{tag}.json'))
    npos, nneg, ns1, nsyn = (int(kv.get(k, v)) for k, v in
                             (('npos', 200000), ('nneg', 200000), ('ns1', 150000), ('nsyn', 100000)))
    # ---- precision of the same cutoffs on labelled tsim
    o = pl.read_parquet(f'{W}/oof_{tag}.parquet').select('id1', 'id2', 'p')
    gt = pl.read_parquet(gt_path('tsim')).select(enc('s1').alias('id1'), enc('id2').alias('id2'))
    pv, nv = pools(o)
    nv = lookalike(attach(nv.sample(min(nv.height, 400000), seed=1), recs('tsim', (1,)), recs('tsim', (2, 3))))
    nv = nv.filter(pl.col('sim') >= 80)
    tp = pv.join(gt, on=['id1', 'id2'], how='semi').height
    fn = nv.join(gt, on=['id1', 'id2'], how='semi').height
    print(f'tsim cutoffs: pos {pv.height} precision {tp / pv.height:.4f} | hard neg (sampled) {nv.height} '
          f'precision {1 - fn / max(nv.height, 1):.4f} (cfg {cfg["method"]})', flush=True)
    del o, pv, nv
    # ---- unseen-country S1 of test
    s1 = pl.read_parquet(f'{W}/n_test_s1.parquet', columns=['id', 'name', 'addr', 'country', 'core', 'ad'])
    uc = unseen(s1)
    s1 = s1.filter(pl.col('country').is_in(uc)).with_columns(enc('id').alias('id'), pl.col('name').fill_null(''),
                                                              pl.col('addr').fill_null(''), pl.col('core').fill_null(''),
                                                              pl.col('ad').fill_null(''))
    print('unseen countries', uc, 'S1', s1.height, flush=True)
    r1 = s1.select('id', 'name', 'addr')
    r2 = recs('test', (2, 3))
    p = pl.read_parquet(f'{W}/pte_{tag}.parquet').select('id1', 'id2', 'p').join(r1.select(pl.col('id').alias('id1')),
                                                                                on='id1', how='semi')
    pos, neg = pools(p)
    neg = lookalike(attach(neg, r1, r2)).filter(pl.col('sim') >= 80)
    print(f'test unseen: scored pairs {p.height}, pos pool {pos.height}, hard-neg pool {neg.height}', flush=True)
    pos = attach(pos.sample(min(pos.height, npos), seed=2), r1, r2)
    neg = neg.sample(min(neg.height, nneg), seed=3)
    rows = [(fmt(a, b), fmt(c, d), 1, 'pos') for a, b, c, d in pos.select('n1', 'a1', 'n2', 'a2').iter_rows()]
    rows += [(fmt(a, b), fmt(c, d), 0, 'neg') for a, b, c, d in neg.select('n1', 'a1', 'n2', 'a2').iter_rows()]
    # ---- exact negatives: distinct S1 records sharing the address (different core) or the core name (other address)
    at = s1.select('id', 'core', 'ad')
    sa = adjacent(at, 'ad').join(at.rename({'id': 'ida', 'core': 'ca', 'ad': 'aa'}), on='ida').join(
        at.rename({'id': 'idb', 'core': 'cb', 'ad': 'ab'}), on='idb').filter(pl.col('ca') != pl.col('cb'))
    sc = adjacent(at, 'core').join(at.rename({'id': 'ida', 'core': 'ca', 'ad': 'aa'}), on='ida').join(
        at.rename({'id': 'idb', 'core': 'cb', 'ad': 'ab'}), on='idb').filter(pl.col('aa') != pl.col('ab'))
    print(f'S1-S1 exact negatives: same address {sa.height}, same core name {sc.height}', flush=True)
    s1n = pl.concat([sa.select('ida', 'idb'), sc.select('ida', 'idb')]).unique()
    s1n = s1n.sample(min(s1n.height, ns1), seed=4)
    s1n = (s1n.join(r1.rename({'id': 'ida', 'name': 'n1', 'addr': 'a1'}), on='ida')
           .join(r1.rename({'id': 'idb', 'name': 'n2', 'addr': 'a2'}), on='idb'))
    for a, b, c, d in s1n.select('n1', 'a1', 'n2', 'a2').iter_rows():
        c, d = noise(c, d)
        rows.append((fmt(a, b), fmt(c, d), 0, 's1neg'))
    for a, b in r1.sample(min(r1.height, nsyn), seed=5).select('name', 'addr').iter_rows():
        c, d = noise(a, b)
        rows.append((fmt(a, b), fmt(c, d), 1, 'syn'))
    RND.shuffle(rows)
    res = pl.DataFrame(rows, schema=['t1', 't2', 'y', 'kind'], orient='row')
    res.write_parquet(f'{W}/{out}.parquet')
    print('wrote', out, res.height, dict(res.group_by('kind').len().iter_rows()), flush=True)
    for k in ('pos', 'neg', 's1neg', 'syn'):
        for t1, t2 in res.filter(pl.col('kind') == k).head(2).select('t1', 't2').iter_rows():
            print(f'  [{k}] {t1}  <>  {t2}')


if __name__ == '__main__':
    main(dict(a.split('=', 1) for a in sys.argv[1:]))
