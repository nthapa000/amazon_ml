"""Metric-aware decision: one-to-one assignment + (global threshold | expected-F0.5 set selection).

python decide.py tune tag=final split=tsim
    -> prints OOF macro F0.5 per method/param on the labelled split, saves the best rule to W/dec_{tag}.json
python decide.py apply tag=final out=DIR [fr=-2]
    -> DIR/matching_results.tsv  (the rule applied to the final model's test scores W/pte_{tag})
       DIR/candidate_pairs.tsv   (every pair in W/pte_{tag}: exactly the pairs the final model scored)
       DIR/test_stats.json
    fr: logit shift for S1 entities whose country never occurs in the training data.
"""
import os
import sys
import json

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
from metric import macro_fbeta_arr  # noqa: E402
from block import enc, dec, gt_path  # noqa: E402

W = os.environ.get('ER_WORK', 'work')
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
    """Pairs with score >= t."""
    return d.filter(pl.col('p') >= t).select('id1', 'id2')


def pick_ef(d, m=0.0, a=1.0, floor=0.02):
    """expected-F: per S1 choose top-k maximizing (1+b2)C_k/(b2(S+m)+k), or empty if P(no match) is higher."""
    b2 = B * B
    d = d.with_columns(q=(pl.col('p') ** a).clip(1e-6, 1 - 1e-6)).filter(pl.col('q') >= floor)
    d = d.sort(['id1', 'q'], descending=[False, True])
    d = d.with_columns(k=pl.col('q').cum_count().over('id1'), C=pl.col('q').cum_sum().over('id1'),
                       S=pl.col('q').sum().over('id1'), e0=(1 - pl.col('q')).log().sum().over('id1').exp())
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
    """Grid over one-to-one margin x (threshold | expected-F parameters) on OOF macro F0.5; saves the best."""
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


def select(d, cfg):
    """Apply a decision rule (the dict saved by tune) to scored pairs -> selected (id1, id2)."""
    d = one2one(d, cfg.get('margin', 0.0))
    if cfg['method'] == 'thr':
        return pick_thr(d, cfg['t'])
    return pick_ef(d, cfg['m'], cfg['a'])


def write_lists(path, s1ids, sel, col):
    """One row per S1 id with its selected ids joined by commas (empty when none)."""
    sel = sel.select(dec('id1').alias('id1'), dec('id2').alias('id2'))
    g = sel.group_by('id1').agg(pl.col('id2').unique().sort().str.join(',').alias(col))
    out = s1ids.join(g.rename({'id1': 'source1_entity_id'}), on='source1_entity_id', how='left').fill_null('')
    out.write_csv(path, separator='\t', quote_style='never')
    return out


def unseen(s1):
    """test countries that never occur in the training data (country is an open set: nothing is named in the code)."""
    seen = set(pl.read_parquet(f'{W}/n_train_s1.parquet', columns=['country'])['country'].unique().to_list())
    return sorted(set(s1['country'].unique().to_list()) - seen)


def apply(tag, outdir, fr=None):
    """fr: optional delta -> p' = sigmoid(logit(p) + delta) for pairs of S1 entities whose country is unseen in training
    (calibration for countries without training labels; delta chosen with leaderboard deltas that differ only there)."""
    os.makedirs(outdir, exist_ok=True)
    cfg = json.load(open(f'{W}/dec_{tag}.json'))
    p = pl.read_parquet(f'{W}/pte_{tag}.parquet')
    cand = p.select('id1', 'id2')                    # the final model's inference set = candidate_pairs.tsv
    s1 = pl.read_parquet(f'{W}/n_test_s1.parquet', columns=['id', 'country']).rename({'id': 'source1_entity_id'})
    shift = {c: fr for c in unseen(s1)} if fr is not None else None
    if shift:
        ctry = s1.select(enc('source1_entity_id').alias('id1'), 'country')
        p = p.join(ctry, on='id1', how='left')
        lg = (pl.col('p').clip(1e-7, 1 - 1e-7) / (1 - pl.col('p').clip(1e-7, 1 - 1e-7))).log()
        for c, dlt in shift.items():
            p = p.with_columns(p=pl.when(pl.col('country') == c).then(1 / (1 + (-(lg + dlt)).exp())).otherwise(pl.col('p')))
        p = p.drop('country')
        print('country shift', shift, flush=True)
    sel = select(p, cfg)
    m = write_lists(f'{outdir}/matching_results.tsv', s1.select('source1_entity_id'), sel, 'matched_entity_ids')
    write_lists(f'{outdir}/candidate_pairs.tsv', s1.select('source1_entity_id'), cand, 'candidate_entity_ids')
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
        apply(kv['tag'], kv['out'], fr=float(kv['fr']) if 'fr' in kv else None)
