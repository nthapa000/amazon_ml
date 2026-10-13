"""Per-country decision rules (plan A1): each country seen in training gets its own decision config, tuned on tsim;
unseen countries keep the global config of dec_<tag>.json (plus the fr shift). No src/ edits.

  tune  tag=<t> [a5=1]                 grid per country on tsim; reports vs the global config: overall, per country,
                                       5 folds, and an honest CV estimate (config picked on 4 folds, scored on the 5th)
                                       -> W/pcdec_<t>.json
  apply tag=<t> out=<new> [fr=-2] [a5=1] [cand=cand_test_final]
        -> $ER_OUT/<new>/{matching_results.tsv, candidate_pairs.tsv, test_stats.json}, W/dec_<new>.json (for snap.py)

a5=1 (plan A5): a record left unassigned by the rule goes to its runner-up S1 when that pair has p >= 0.9.
Blocking is within country, so a record's candidate S1s all share one country and per-country rules never compete.
"""
import os
import sys
import json

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from block import enc, gt_path  # noqa: E402
from decide import one2one, pick_thr, pick_ef, score, write_lists, unseen  # noqa: E402
from metric import macro_fbeta_arr  # noqa: E402
from qfix import shift, moments  # noqa: E402

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')
OUT = os.environ.get('ER_OUT', '/scratch/m25csa019/er_out')
MARGINS = (0.15, 0.2, 0.25, 0.3)


def grid(base):
    cfgs = [dict(method='thr', margin=mg, t=t) for mg in MARGINS for t in (0.5, 0.6, 0.7)]
    cfgs += [dict(method='ef', margin=mg, m=base.get('m', 0.0), a=a, floor=fl)
             for mg in MARGINS for a in (0.9, 1.0, 1.1, 1.25) for fl in (0.01, 0.02, 0.05)]
    return cfgs


def pick(d1, cfg):
    """d1 = one2one output for cfg['margin']."""
    if cfg['method'] == 'thr':
        return pick_thr(d1, cfg['t'])
    return pick_ef(d1, cfg['m'], cfg['a'], floor=cfg.get('floor', 0.02))


def a5(sel, p, thr=0.9):
    """unassigned records -> runner-up S1 if that pair has p >= thr."""
    r = (p.sort(['id2', 'p'], descending=[False, True]).with_columns(r=pl.col('p').cum_count().over('id2'))
         .filter((pl.col('r') == 2) & (pl.col('p') >= thr)).select('id1', 'id2'))
    r = r.join(sel.select('id2'), on='id2', how='anti')
    return pl.concat([sel.select('id1', 'id2'), r])


def per_s1(sel, s1, gt):
    """-> numpy F0.5 per S1, aligned with s1 rows."""
    tp = sel.join(gt.rename({'s1': 'id1'}), on=['id1', 'id2'], how='semi')
    per = (s1.select('s1').join(gt.group_by('s1').len('nt'), on='s1', how='left')
           .join(sel.group_by('id1').len('npred').rename({'id1': 's1'}), on='s1', how='left')
           .join(tp.group_by('id1').len('ntp').rename({'id1': 's1'}), on='s1', how='left').fill_null(0))
    return macro_fbeta_arr(per['nt'], per['npred'], per['ntp'], 0.5)[1]


def select_pc(p, s1, cfgs, glob, do_a5=False):
    """p: (id1,id2,p); s1: (s1,country); cfgs: {country: cfg}; other countries use glob."""
    d = p.select('id1', 'id2', 'p').with_columns(pmax=pl.col('p').max().over('id2'), p2=pl.col('p').top_k(2).min().over('id2'),
                                                 n2=pl.len().over('id2'))
    d = d.filter(pl.col('p') == pl.col('pmax')).unique('id2', keep='first')      # = decide.one2one, competition global
    d = d.join(s1.rename({'s1': 'id1'}), on='id1', how='left')
    out = []
    for c in d['country'].unique().to_list():
        cfg = cfgs.get(c, glob)
        x = d.filter(pl.col('country') == c)
        if cfg.get('margin', 0.0) > 0:
            x = x.filter((pl.col('n2') == 1) | (pl.col('p') - pl.col('p2') >= cfg['margin']))
        out.append(pick(x.select('id1', 'id2', 'p'), cfg).select('id1', 'id2'))
    sel = pl.concat(out)
    return a5(sel, p) if do_a5 else sel


def cmd_tune(kv):
    tag = kv['tag']
    glob = json.load(open(f'{W}/dec_{tag}.json'))
    s1 = pl.read_parquet(f'{W}/n_tsim_s1.parquet', columns=['id', 'country']).select(enc('id').alias('s1'), 'country')
    g = pl.read_parquet(gt_path('tsim')).select(enc('s1').alias('s1'), enc('id2').alias('id2'))
    o = pl.read_parquet(f'{W}/oof_{tag}.parquet', columns=['id1', 'id2', 'p'])
    fold = (s1['s1'].hash(seed=7) % 5).to_numpy()
    ctry = s1['country'].to_numpy()
    cs = sorted(set(ctry))
    base_cfg = {k: v for k, v in glob.items() if k in ('method', 'margin', 't', 'm', 'a', 'floor')}
    cfgs = [base_cfg] + [c for c in grid(glob) if c != base_cfg]
    F = np.zeros((len(cfgs), s1.height))
    d1 = {}
    for i, cfg in enumerate(cfgs):
        if cfg['margin'] not in d1:
            d1[cfg['margin']] = one2one(o, cfg['margin'])
        F[i] = per_s1(pick(d1[cfg['margin']], cfg), s1, g)
        if i == 0:
            print(f'base reproduces dec_{tag}.json: {F[0].mean():.6f} vs {glob["oof"]["ALL"]:.6f}', flush=True)
        if i % 10 == 0:
            print(f'  {i}/{len(cfgs)} configs', flush=True)

    def best(mask):                                  # config index maximizing mean F over the masked S1
        return int(np.argmax(F[:, mask].mean(1)))

    choice = {c: best(ctry == c) for c in cs}
    comb = np.zeros(s1.height)
    for c in cs:
        comb[ctry == c] = F[choice[c], ctry == c]
    cv = np.zeros(s1.height)                         # honest: choose on the other 4 folds
    for k in range(5):
        for c in cs:
            m_tr, m_te = (ctry == c) & (fold != k), (ctry == c) & (fold == k)
            cv[m_te] = F[best(m_tr), m_te]
    res = dict(base=float(F[0].mean()), comb=float(comb.mean()), cv=float(cv.mean()), cfg={c: cfgs[choice[c]] for c in cs})
    print(f'\nglobal {F[0].mean():.6f} | per-country (in-sample) {comb.mean():.6f} ({comb.mean() - F[0].mean():+.6f}) | '
          f'per-country CV {cv.mean():.6f} ({cv.mean() - F[0].mean():+.6f})', flush=True)
    for c in cs:
        m = ctry == c
        print(f'  {c:7s} global {F[0, m].mean():.6f}  best {comb[m].mean():.6f} ({comb[m].mean() - F[0, m].mean():+.6f})  '
              f'CV {cv[m].mean():.6f} ({cv[m].mean() - F[0, m].mean():+.6f})  cfg {cfgs[choice[c]]}', flush=True)
    fd = [comb[fold == k].mean() - F[0, fold == k].mean() for k in range(5)]
    fcv = [cv[fold == k].mean() - F[0, fold == k].mean() for k in range(5)]
    print('  folds in-sample ' + ' '.join(f'{x:+.6f}' for x in fd) + f'  ({sum(x > 0 for x in fd)}/5 +)', flush=True)
    print('  folds CV        ' + ' '.join(f'{x:+.6f}' for x in fcv) + f'  ({sum(x > 0 for x in fcv)}/5 +)', flush=True)
    res.update(folds=fd, folds_cv=fcv)
    if kv.get('a5') == '1':                          # A5 on top of the chosen per-country configs
        sel = select_pc(o, s1, res['cfg'], base_cfg, do_a5=False)
        sel5 = a5(sel, o)
        f0, f5 = per_s1(sel, s1, g), per_s1(sel5, s1, g)
        d5 = [f5[fold == k].mean() - f0[fold == k].mean() for k in range(5)]
        print(f'  A5 runner-up>=0.9: {f5.mean() - f0.mean():+.6f}  links +{sel5.height - sel.height}  folds '
              + ' '.join(f'{x:+.6f}' for x in d5), flush=True)
        res['a5'] = dict(gain=float(f5.mean() - f0.mean()), folds=d5)
    ok = (res['cv'] > res['base'] and sum(x > 0 for x in fcv) >= 3
          and all(cv[ctry == c].mean() >= F[0, ctry == c].mean() for c in cs))
    res['adopt'] = bool(ok)
    print('GATE (CV > global, >= 3/5 CV folds +, no country worse):', 'PASS' if ok else 'FAIL', flush=True)
    json.dump(res, open(f'{W}/pcdec_{tag}.json', 'w'), indent=1)


def cmd_apply(kv):
    tag, out = kv['tag'], kv['out']
    glob = json.load(open(f'{W}/dec_{tag}.json'))
    base_cfg = {k: v for k, v in glob.items() if k in ('method', 'margin', 't', 'm', 'a', 'floor')}
    pc = json.load(open(f'{W}/pcdec_{kv.get("from", tag)}.json'))['cfg']
    do5 = kv.get('a5') == '1'
    s1v = pl.read_parquet(f'{W}/n_tsim_s1.parquet', columns=['id', 'country']).select(enc('id').alias('s1'), 'country')
    g = pl.read_parquet(gt_path('tsim')).select(enc('s1').alias('s1'), enc('id2').alias('id2'))
    oof = score(select_pc(pl.read_parquet(f'{W}/oof_{tag}.parquet', columns=['id1', 'id2', 'p']), s1v, pc, base_cfg, do5),
                s1v, g)
    print('tsim', json.dumps(oof), flush=True)
    s1 = pl.read_parquet(f'{W}/n_test_s1.parquet', columns=['id', 'country']).select(enc('id').alias('s1'), 'country')
    seen = {c: v for c, v in pc.items() if c not in unseen(s1.rename({'s1': 'id'}))}
    fr = float(kv['fr']) if 'fr' in kv else None
    p = shift(pl.read_parquet(f'{W}/pte_{tag}.parquet', columns=['id1', 'id2', 'p']), s1, fr=fr)
    sel = select_pc(p, s1, seen, base_cfg, do5)
    cand = pl.read_parquet(f'{W}/{kv.get("cand", "cand_test_final")}.parquet', columns=['id1', 'id2'])
    assert sel.join(cand, on=['id1', 'id2'], how='anti').height == 0, 'every match must be a candidate'
    od = f'{OUT}/{out}'
    os.makedirs(od, exist_ok=True)
    ids = pl.read_parquet(f'{W}/n_test_s1.parquet', columns=['id']).rename({'id': 'source1_entity_id'})
    write_lists(f'{od}/matching_results.tsv', ids, sel, 'matched_entity_ids')
    write_lists(f'{od}/candidate_pairs.tsv', ids, cand, 'candidate_entity_ids')
    m = moments(sel, s1)
    print(m, flush=True)
    json.dump(dict(links=sel.height, rows=ids.height, pred_empty=dict(zip(m['country'], m['empty']))),
              open(f'{od}/test_stats.json', 'w'))
    json.dump(dict(base_cfg, oof=oof, split='tsim', base=tag, per_country=seen, fr=fr, a5=do5),
              open(f'{W}/dec_{out}.json', 'w'))


if __name__ == '__main__':
    c = sys.argv[1]
    kv = dict(a.split('=', 1) for a in sys.argv[2:])
    dict(tune=cmd_tune, apply=cmd_apply)[c](kv)
