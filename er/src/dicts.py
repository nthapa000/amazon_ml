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


if __name__ == '__main__':
    main()
