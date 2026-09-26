"""Label-free self-check per country (read-only): how good does the model think its own selection is?

For a calibrated model, the probabilities give unbiased estimates without labels:
- own_rate : mean over S2/S3 records of sum_S1 p (expected share of records that have a true owner)
- viol     : share of S2/S3 records with sum p > 1 (impossible under one-to-one -> miscalibration signal)
- prec_est : mean p of the selected links (expected precision)
- F_est    : expected macro F0.5 of the selection, per S1 via ratio of expectations:
             empty prediction -> prod(1 - p); otherwise 1.25*sum_sel(p) / (0.25*sum_all(p') + k),
             p' = p divided by max(1, sum p of its S2/S3 record) (one-to-one consistent)
On tsim the same numbers are printed next to the TRUE F, which shows how far F_est can be trusted.
Beliefs are the model's RAW p; the selection is the uploaded one (test: unseen-country shift fr).

python ana/selfcheck.py tags=fin3c+fin4c fr=-2
"""
import os
import sys
import json

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from block import enc, gt_path  # noqa: E402
from decide import select, score, unseen  # noqa: E402

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')


def shifted(p, s1, fr):
    if fr is None:
        return p
    cs = unseen(s1.rename({'s1': 'id'}))
    lg = (pl.col('p').clip(1e-7, 1 - 1e-7) / (1 - pl.col('p').clip(1e-7, 1 - 1e-7))).log()
    return p.with_columns(p=pl.when(pl.col('country').is_in(cs)).then(1 / (1 + (-(lg + fr)).exp())).otherwise(pl.col('p')))


def check(p, sel, s1):
    """p: (id1,id2,p,country) raw beliefs; sel: selected (id1,id2); s1: (s1,country) -> per-country dict."""
    rec = p.group_by('id2').agg(pl.col('p').sum().alias('sp'), pl.col('country').first())
    p = p.join(rec.select('id2', 'sp'), on='id2').with_columns(pn=pl.col('p') / pl.max_horizontal(pl.col('sp'), pl.lit(1.0)))
    ps = p.join(sel.with_columns(s=pl.lit(1)), on=['id1', 'id2'], how='left').with_columns(pl.col('s').fill_null(0))
    ent = ps.group_by('id1').agg(T=pl.col('pn').sum(), TP=(pl.col('p') * pl.col('s')).sum(), k=pl.col('s').sum(),
                                 e0=(1 - pl.col('p').clip(0, 1 - 1e-9)).log().sum().exp())
    ent = s1.rename({'s1': 'id1'}).join(ent, on='id1', how='left').with_columns(
        pl.col('T').fill_null(0), pl.col('TP').fill_null(0), pl.col('k').fill_null(0), pl.col('e0').fill_null(1.0))
    ent = ent.with_columns(F=pl.when(pl.col('k') == 0).then(pl.col('e0'))
                           .otherwise(1.25 * pl.col('TP') / (0.25 * pl.col('T') + pl.col('k'))))
    selp = ps.filter(pl.col('s') == 1)
    out = {}
    for c in sorted(s1['country'].unique().to_list()) + ['ALL']:
        e = ent if c == 'ALL' else ent.filter(pl.col('country') == c)
        r = rec if c == 'ALL' else rec.filter(pl.col('country') == c)
        sp = selp if c == 'ALL' else selp.filter(pl.col('country') == c)
        out[c] = dict(n_s1=e.height, F_est=float(e['F'].mean()), prec_est=float(sp['p'].mean()),
                      own_rate=float(r['sp'].mean()), viol=float((r['sp'] > 1.0).mean()),
                      links_per_s1=sp.height / e.height, pred_empty=float((e['k'] == 0).mean()))
    return out


def run(tag, fr):
    cfg = json.load(open(f'{W}/dec_{tag}.json'))
    res = {}
    for split, pf in (('tsim', 'oof'), ('test', 'pte')):
        s1 = pl.read_parquet(f'{W}/n_{split}_s1.parquet', columns=['id', 'country']).select(enc('id').alias('s1'), 'country')
        p = pl.read_parquet(f'{W}/{pf}_{tag}.parquet').select('id1', 'id2', 'p')
        p = p.join(s1.rename({'s1': 'id1'}), on='id1', how='left')
        sel = select(shifted(p, s1, fr if split == 'test' else None).select('id1', 'id2', 'p'), cfg)
        res[split] = check(p, sel, s1)
        if split == 'tsim':
            gt = pl.read_parquet(gt_path(split)).select(enc('s1').alias('s1'), enc('id2').alias('id2'))
            true = score(sel, s1, gt)
            for c in res[split]:
                res[split][c]['F_true'] = true.get(c)
    return res


if __name__ == '__main__':
    kv = dict(a.split('=', 1) for a in sys.argv[1:])
    fr = float(kv['fr']) if 'fr' in kv else None
    allr = {}
    for tag in kv['tags'].replace('+', ',').split(','):
        allr[tag] = r = run(tag, fr)
        for split in ('tsim', 'test'):
            print(f'\n== {tag} {split} (fr={fr if split == "test" else None})', flush=True)
            for c, d in r[split].items():
                print(f'  {c:7s} ' + '  '.join(f'{k}={v:.5f}' if isinstance(v, float) else f'{k}={v}' for k, v in d.items()),
                      flush=True)
    json.dump(allr, open(f'{W}/selfcheck_{kv["tags"].replace(",", "_").replace("+", "_")}.json', 'w'), indent=1)
