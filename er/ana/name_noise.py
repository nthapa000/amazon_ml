"""How do TRUE matches differ in the core name? (labelled US/India train pairs)  -> rates for the synthetic French matches.

For a sample of ground-truth (S1, S2/S3) pairs, compare the token lists of the normalized core names and count, per country:
  equal | S2/S3 keeps only the first words (truncation) | S2/S3 drops words elsewhere | S2/S3 adds words |
  one word replaced (typo-like: rapidfuzz ratio >= 80, or a different word) | same words in another order | other
Also prints examples of the replaced-word case, so the synthetic noise can copy what the real noise does.
python ana/name_noise.py [n=400000]      (ER_WORK = work dir)
"""
import os
import sys
import json
import random
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import polars as pl  # noqa: E402
from rapidfuzz import fuzz  # noqa: E402
from block import enc  # noqa: E402

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')
kv = dict(a.split('=', 1) for a in sys.argv[1:])
n = int(kv.get('n', 400000))


def kind(t1, t2):
    if t1 == t2:
        return 'equal'
    if not t1 or not t2:
        return 'empty'
    if len(t2) < len(t1) and t1[:len(t2)] == t2:
        return 'trunc'                                   # S2/S3 = first words of S1
    it = iter(t1)
    if len(t2) < len(t1) and all(w in it for w in t2):
        return 'drop'                                    # S2/S3 = S1 without some (non-trailing) words
    it = iter(t2)
    if len(t1) < len(t2) and all(w in it for w in t1):
        return 'add'                                     # S2/S3 = S1 plus extra words
    if len(t1) == len(t2):
        if sorted(t1) == sorted(t2):
            return 'reorder'
        d = [(a, b) for a, b in zip(t1, t2) if a != b]
        if len(d) == 1:
            r = fuzz.ratio(*d[0])
            return 'sub_typo' if r >= 80 else ('sub_garble' if r >= 50 else 'sub_word')   # sub_word = a genuinely different word
    return 'other'


gt = pl.read_parquet(f'{W}/gt_pairs.parquet').select(enc('s1').alias('id1'), enc('id2').alias('id2')).sample(n, seed=1)
s1 = pl.read_parquet(f'{W}/n_train_s1.parquet', columns=['id', 'country', 'core']).select(
    enc('id').alias('id1'), 'country', pl.col('core').alias('c1'))
s23 = pl.concat([pl.read_parquet(f'{W}/n_train_s{k}.parquet', columns=['id', 'core']) for k in (2, 3)]).select(
    enc('id').alias('id2'), pl.col('core').alias('c2'))
d = gt.join(s1, on='id1').join(s23, on='id2').with_columns(pl.col('c1').fill_null(''), pl.col('c2').fill_null(''))
print('pairs', d.height, flush=True)

cnt, ex = {}, {}
rep, pos = {}, {}
random.seed(3)
for c, a, b in d.select('country', 'c1', 'c2').iter_rows():
    k = kind(a.split(), b.split())
    cnt.setdefault(c, Counter())[k] += 1
    if k == 'sub_word':
        t1, t2 = a.split(), b.split()
        i = next(j for j in range(len(t1)) if t1[j] != t2[j])
        rep.setdefault(c, Counter())[(t1[i], t2[i])] += 1
        pos.setdefault(c, Counter())['last' if i == len(t1) - 1 else 'first' if i == 0 else 'middle'] += 1
    if k in ('sub_word', 'sub_garble', 'add', 'reorder', 'other') and random.random() < 0.0004:
        ex.setdefault((c, k), []).append((a, b))
out = {}
for c, cc in sorted(cnt.items()):
    tot = sum(cc.values())
    out[c] = {k: round(v / tot, 5) for k, v in cc.most_common()}
    print(c, tot, out[c], flush=True)
for (c, k), v in sorted(ex.items()):
    print(f'-- {c} {k}')
    for a, b in v[:8]:
        print(f'   S1 "{a}"  <>  S2/S3 "{b}"')
for c in rep:
    print(f'-- {c} genuinely-different-word swaps: position {dict(pos[c])}; top replaced->new pairs {rep[c].most_common(12)}')
    new = Counter()
    for (x, y), v in rep[c].items():
        new[y] += v
    print(f'   distinct new words {len(new)}; top new words {new.most_common(12)}')
json.dump(out, open(f'{W}/name_noise_rates.json', 'w'), indent=1)
