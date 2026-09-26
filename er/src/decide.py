"""Metric-aware decision: one-to-one assignment + (global threshold | expected-F0.5 set selection).

python decide.py tune tag=s1            -> prints OOF macro F0.5 per method/param, saves W/dec_{tag}.json
python decide.py apply tag=s1 out=DIR   -> DIR/matching_results.tsv (+ candidate_pairs.tsv from W/cand_test)
"""
import os
import sys
import json

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
from metric import macro_fbeta_arr  # noqa: E402
from block import enc, dec, gt_path  # noqa: E402

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')
B = 0.5


def one2one(d, margin=0.0):
    """keep, for each S2/S3 id, only its best S1 (optionally require margin over the runner-up)."""
    d = d.with_columns(pmax=pl.col('p').max().over('id2'),
                       p2=pl.col('p').top_k(2).min().over('id2'), n2=pl.len().over('id2'))
    d = d.filter(pl.col('p') == pl.col('pmax'))
    d = d.unique('id2', keep='first')
    if margin > 0:
        d = d.filter((pl.col('n2') == 1) | (pl.col('p') - pl.col('p2') >= margin))
    return d.drop('pmax', 'p2', 'n2')


def pick_thr(d, t):
    return d.filter(pl.col('p') >= t).select('id1', 'id2')


def pick_ef(d, m=0.0, a=1.0, floor=0.02, gate=None, g=1.0):
    """expected-F: per S1 choose top-k maximizing (1+b2)C_k/(b2(S+m)+k), or empty if P(no match) is higher.
    gate: optional (id1, q0) entity-level P(no match) (gate.py); replaces the independence estimate prod(1-p)."""
    b2 = B * B
    d = d.with_columns(q=(pl.col('p') ** a).clip(1e-6, 1 - 1e-6)).filter(pl.col('q') >= floor)
    d = d.sort(['id1', 'q'], descending=[False, True])
    d = d.with_columns(k=pl.col('q').cum_count().over('id1'), C=pl.col('q').cum_sum().over('id1'),
                       S=pl.col('q').sum().over('id1'), e0=(1 - pl.col('q')).log().sum().over('id1').exp())
    if gate is not None:
        d = d.join(gate.select('id1', 'q0'), on='id1', how='left').with_columns(
            e0=pl.coalesce(pl.col('q0') * g, pl.col('e0'))).drop('q0')
    d = d.with_columns(ef=(1 + b2) * pl.col('C') / (b2 * (pl.col('S') + m) + pl.col('k')))
    d = d.with_columns(efm=pl.col('ef').max().over('id1'))
    kbest = d.filter(pl.col('ef') == pl.col('efm')).group_by('id1').agg(pl.col('k').min().alias('kb'))
    d = d.join(kbest, on='id1').filter((pl.col('k') <= pl.col('kb')) & (pl.col('efm') > pl.col('e0')))
    return d.select('id1', 'id2')


def score(sel, s1, gt, by_country=True):
    """sel: predicted (id1,id2). s1: (s1, country). gt: (s1,id2). -> dict country->F."""
    tp = sel.join(gt.rename({'s1': 'id1'}), on=['id1', 'id2'], how='semi')
    per = (s1.join(gt.group_by('s1').len('nt'), on='s1', how='left')
           .join(sel.group_by('id1').len('npred').rename({'id1': 's1'}), on='s1', how='left')
           .join(tp.group_by('id1').len('ntp').rename({'id1': 's1'}), on='s1', how='left').fill_null(0))
    out = {'ALL': macro_fbeta_arr(per['nt'], per['npred'], per['ntp'], B)[0]}
    if by_country:
        for c in per['country'].unique().sort():
            g = per.filter(pl.col('country') == c)
            out[c] = macro_fbeta_arr(g['nt'], g['npred'], g['ntp'], B)[0]
        out['F0.4'] = macro_fbeta_arr(per['nt'], per['npred'], per['ntp'], 0.4)[0]
        out['pred_empty'] = float((per['npred'] == 0).mean())
    return out


def tune(tag, split='train'):
    o = pl.read_parquet(f'{W}/oof_{tag}.parquet').select('id1', 'id2', 'p')
    s1 = pl.read_parquet(f'{W}/n_{split}_s1.parquet', columns=['id', 'country']).select(enc('id').alias('s1'), 'country')
    gt = pl.read_parquet(gt_path(split)).select(enc('s1').alias('s1'), enc('id2').alias('id2'))
    best = (-1, None)
    for mg in (0.0, 0.1, 0.2, 0.3):
        d = one2one(o, mg)
        for t in (0.6, 0.7, 0.8):
            r = score(pick_thr(d, t), s1, gt, by_country=False)
            print(f'thr  o2o={mg} t={t}: {r["ALL"]:.5f}', flush=True)
            if r['ALL'] > best[0]:
                best = (r['ALL'], dict(method='thr', margin=mg, t=t))
        for m in (0.0, 0.5, 1.0):
            for a in (1.0, 1.25, 1.5):
                r = score(pick_ef(d, m, a), s1, gt, by_country=False)
                print(f'ef   o2o={mg} m={m} a={a}: {r["ALL"]:.5f}', flush=True)
                if r['ALL'] > best[0]:
                    best = (r['ALL'], dict(method='ef', margin=mg, m=m, a=a))
    # no one-to-one baseline for reference
    r = score(pick_thr(o, 0.5), s1, gt, by_country=False)
    print(f'thr  no-o2o t=0.5: {r["ALL"]:.5f}')
    cfg = best[1]
    sc = score(select(o, cfg), s1, gt)
    print('BEST', cfg, json.dumps(sc), flush=True)
    json.dump(dict(cfg, oof=sc, split=split), open(f'{W}/dec_{tag}.json', 'w'))


def select(d, cfg, gate=None):
    d = one2one(d, cfg.get('margin', 0.0))
    if cfg['method'] == 'thr':
        return pick_thr(d, cfg['t'])
    return pick_ef(d, cfg['m'], cfg['a'], gate=gate if cfg.get('gate') else None, g=cfg.get('g', 1.0))


def write_lists(path, s1ids, sel, col):
    sel = sel.select(dec('id1').alias('id1'), dec('id2').alias('id2'))
    g = sel.group_by('id1').agg(pl.col('id2').unique().sort().str.join(',').alias(col))
    out = s1ids.join(g.rename({'id1': 'source1_entity_id'}), on='source1_entity_id', how='left').fill_null('')
    out.write_csv(path, separator='\t', quote_style='never')
    return out


def apply(tag, outdir, cand, shift=None, nocand=False):
    """shift: optional {country: delta} -> p' = sigmoid(logit(p) + delta) for pairs of that country's S1 entities
    (country-level calibration for a country without training labels; chosen with country-only leaderboard deltas)."""
    os.makedirs(outdir, exist_ok=True)
    cfg = json.load(open(f'{W}/dec_{tag}.json'))
    p = pl.read_parquet(f'{W}/pte_{tag}.parquet')
    s1 = pl.read_parquet(f'{W}/n_test_s1.parquet', columns=['id', 'country']).rename({'id': 'source1_entity_id'})
    if shift:
        ctry = s1.select(enc('source1_entity_id').alias('id1'), 'country')
        p = p.join(ctry, on='id1', how='left')
        lg = (pl.col('p').clip(1e-7, 1 - 1e-7) / (1 - pl.col('p').clip(1e-7, 1 - 1e-7))).log()
        for c, dlt in shift.items():
            p = p.with_columns(p=pl.when(pl.col('country') == c).then(1 / (1 + (-(lg + dlt)).exp())).otherwise(pl.col('p')))
        p = p.drop('country')
        print('country shift', shift, flush=True)
    gate = pl.read_parquet(f'{W}/{cfg["gate"]}_test.parquet') if cfg.get('gate') else None
    sel = select(p, cfg, gate)
    m = write_lists(f'{outdir}/matching_results.tsv', s1.select('source1_entity_id'), sel, 'matched_entity_ids')
    if not nocand:
        c = pl.read_parquet(f'{W}/{cand}.parquet', columns=['id1', 'id2'])
        write_lists(f'{outdir}/candidate_pairs.tsv', s1.select('source1_entity_id'), c, 'candidate_entity_ids')
    e = s1.join(m, on='source1_entity_id').with_columns(emp=pl.col('matched_entity_ids') == '')
    print('cfg', cfg, 'rows', m.height, 'links', sel.height)
    ce = e.group_by('country').agg(pl.col('emp').mean().alias('pred_empty'), pl.len())
    print(ce)
    json.dump(dict(links=sel.height, rows=m.height, pred_empty=dict(zip(ce['country'], ce['pred_empty']))),
              open(f'{outdir}/test_stats.json', 'w'))


if __name__ == '__main__':
    cmd = sys.argv[1]
    kv = dict(a.split('=', 1) for a in sys.argv[2:])
    if cmd == 'tune':
        tune(kv['tag'], kv.get('split', 'train'))
    else:
        apply(kv['tag'], kv['out'], kv.get('cand', 'cand_test'),
              shift={'France': float(kv['fr'])} if 'fr' in kv else None, nocand=kv.get('nocand') == '1')
