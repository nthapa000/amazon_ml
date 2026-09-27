"""Mine a native-script -> Latin token dictionary from matched TRAIN pairs only.

Names: positional alignment of equal-length token lists (S2/S3 native vs S1 English).
Addresses: Dice co-occurrence between native tokens and S1 address tokens.
Output: W/tdict.json  {native_token: latin_string}
"""
import os
import sys
import json
from collections import Counter, defaultdict

import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
import norm  # noqa: E402

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')


def lat_toks(s):
    return [norm.latin(t) for t in norm.clean_name_toks(norm.split_name(s)[0])]


def addr_toks(s):
    s = norm.clean_addr(s)
    import re
    return [t for t in re.split(r'[\s,.;:()\[\]{}"\-/]+', s) if t]


def main(min_n=2, min_p=0.5):
    s1 = pl.concat([pl.read_parquet(f'{W}/raw_train_s1.parquet')])
    s23 = pl.concat([pl.read_parquet(f'{W}/raw_train_s{k}.parquet') for k in (2, 3)])
    s23 = s23.filter(pl.col('country') == 'India')
    s23 = s23.filter(pl.col('name').str.contains(r'[ऀ-෿]') | pl.col('addr').str.contains(r'[ऀ-෿]'))
    pr = pl.read_parquet(f'{W}/gt_pairs.parquet')
    df = (pr.join(s23.rename({'id': 'id2', 'name': 'n2', 'addr': 'a2'}), on='id2')
          .join(s1.rename({'id': 's1', 'name': 'n1', 'addr': 'a1'}).drop('country'), on='s1'))
    print('pairs with native script', df.shape, flush=True)

    # ---- names: positional alignment
    cnt = defaultdict(Counter)
    for n1, n2 in zip(df['n1'], df['n2']):
        t2 = norm.clean_name_toks(norm.split_name(n2)[0])
        if not any(norm.has_ind(t) for t in t2):
            continue
        t1 = lat_toks(n1)
        if len(t1) != len(t2):
            continue
        for a, b in zip(t2, t1):
            if norm.has_ind(a):
                cnt[a][b] += 1
    td = {}
    for a, c in cnt.items():
        b, n = c.most_common(1)[0]
        tot = sum(c.values())
        if n >= min_n and n / tot >= min_p:
            td[a] = b
    print('name dict', len(td), 'of', len(cnt), flush=True)

    # ---- addresses: Dice
    co, cn, ce = Counter(), Counter(), Counter()
    for a1, a2 in zip(df['a1'], df['a2']):
        t2 = {t for t in addr_toks(a2) if norm.has_ind(t)}
        if not t2:
            continue
        t1 = {norm.latin(t) for t in addr_toks(a1) if not any(ch.isdigit() for ch in t)}
        for a in t2:
            cn[a] += 1
            for b in t1:
                co[(a, b)] += 1
        for b in t1:
            ce[b] += 1
    best = {}
    for (a, b), n in co.items():
        d = 2 * n / (cn[a] + ce[b])
        if n >= min_n and d > best.get(a, (0, ''))[0]:
            best[a] = (d, b)
    na = 0
    for a, (d, b) in best.items():
        if d >= 0.3 and a not in td:
            td[a] = b
            na += 1
    print('addr dict added', na, flush=True)
    json.dump(td, open(f'{W}/tdict.json', 'w'), ensure_ascii=False)
    sm = list(td.items())[:40]
    print(sm)


BUILD_BAD = {'etage', 'batiment', 'bat', 'appartement', 'appt', 'app', 'rez de chaussee', 'rdc', 'escalier', 'esc',
             'maison des associations', 'immeuble'}


def _city_tok(comps):
    """last comma-component that isn't a building/floor descriptor and has no digits (ana/fr_state_gap.py, checked
    against real French test addresses: recovers 88.5% of the S2/S3 state-missing rows)."""
    for x in reversed(comps):
        if x and not any(ch.isdigit() for ch in x) and x not in BUILD_BAD and not any(b in x for b in BUILD_BAD):
            return x
    return ''


def learn_city_region(min_support=3, min_purity=0.9):
    """City -> region lookup for countries with no hand-written state table (e.g. France), learned ONLY from that
    country's own S1 records: S1 already has `state` correct ~100% of the time (norm.py's FR_REGIONS/FR_DEPT
    logic); the city is read from the SAME S1 record's already-normalized `ad` field (comma components), taking
    the last non-digit, non-building-descriptor one. No cross-record inference, no labels used.

    This is exactly the `cities` step already proven inside the old (removed, never committed to git -- recovered
    from /scratch/m25csa019/versions/c5/code.tar.gz) learn_geo(): there it was joined onto S2/S3 and then only used
    as scaffolding to learn a narrower department->region table (which the hand-written FR_DEP already covered at
    99.99%+, hence that lever's honest measured-zero). The city match itself was never persisted or applied on its
    own -- that's the gap this fills: 35% of French S2/S3 records have no department/region text at all, only a
    bare city name, so FR_DEP can never fire on them regardless of how good it is.
    -> W/fr_city_region.json {country: {city: region}}"""
    out, lut_all = {}, {}
    for sp in ('train', 'test'):
        fp = f'{W}/n_{sp}_s1.parquet'
        if not os.path.exists(fp):
            continue
        d = pl.read_parquet(fp, columns=['country', 'ad', 'state'])
        for c in d['country'].unique().sort().to_list():
            if norm.street_map(c)[1]:
                continue
            x = d.filter((pl.col('country') == c) & (pl.col('state').fill_null('') != ''))
            if x.height == 0:
                continue
            cnt = lut_all.setdefault(c, defaultdict(Counter))
            for ad, st in zip(x['ad'].fill_null('').to_list(), x['state'].to_list()):
                comps = [p.strip() for p in ad.split(' , ') if p.strip()]
                city = _city_tok(comps)
                if city and city != st:
                    cnt[city][st] += 1
    for c, cnt in lut_all.items():
        lut = {}
        for city, cc in cnt.items():
            tot = sum(cc.values())
            reg, n = cc.most_common(1)[0]
            if tot >= min_support and n / tot >= min_purity:
                lut[city] = reg
        out[c] = lut
        print(c, f'learned city->region: {len(lut)} entries (from {sum(sum(cc.values()) for cc in cnt.values())} '
              f'region-bearing S1 addresses with a candidate city component)', flush=True)
        for city, reg in sorted(lut.items(), key=lambda kv: -sum(cnt[kv[0]].values()))[:15]:
            print(f'   {city!r} -> {reg!r}  (n={sum(cnt[city].values())}, purity={cnt[city][reg] / sum(cnt[city].values()):.3f})')
    json.dump(out, open(f'{W}/fr_city_region.json', 'w'), ensure_ascii=False)
    return out


if __name__ == '__main__':
    main()

