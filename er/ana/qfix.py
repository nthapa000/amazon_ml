"""Quick decision-layer fixes on an already-scored tag; each result is written as its own version (no src/ edits).

  tune  tag=<t>                        tsim (labelled): base cfg vs per-source caps / count prior, per country + fold
  diag  tag=<t> fr=-3:-2:-1:0          test (label-free): per-country, per-source link moments for each unseen-country
                                       shift; unseen countries should match the seen countries' moments
  apply tag=<t> out=<new> [caps=1] [prior=<lambda>] [fr=d] [fr2=d] [fr3=d] [cand=cand_test_final]
        -> $ER_OUT/<new>/{matching_results.tsv, candidate_pairs.tsv, test_stats.json}, W/dec_<new>.json (for snap.py)

caps : at most CAP[s] selected links per S1 from source s (maxima of the train ground truth).
prior: count prior. The per-S1 Poisson-binomial PMF of the number of true matches K (from the one-to-one q) is
       reweighted by (prior(k) / base_c(k))^lambda. prior = train ground-truth histogram of matches per S1;
       base_c = the model's mean implied PMF over the S1 of country c in the same split (label-free). The posterior
       replaces P(no match) = prod(1-q) and the expected count S = sum q in the expected-F0.5 set rule.
fr / fr2 / fr3: logit shift for S1 of countries unseen in training (all pairs / S2 pairs / S3 pairs).
"""
import os
import sys
import json

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from block import enc, gt_path  # noqa: E402
from decide import one2one, score, write_lists, unseen, B  # noqa: E402

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')
OUT = os.environ.get('ER_OUT', '/scratch/m25csa019/er_out')
KMAX = 12
FLOOR = 0.02
SRC = pl.col('id2') // 10_000_000_000


def s1tab(split):
    return pl.read_parquet(f'{W}/n_{split}_s1.parquet', columns=['id', 'country']).select(enc('id').alias('s1'), 'country')


def gt(split):
    return pl.read_parquet(gt_path(split)).select(enc('s1').alias('s1'), enc('id2').alias('id2'))


def train_stats():
    """train ground truth: matches-per-S1 histogram (0..KMAX, last bin = KMAX+) and per-source maxima / constants."""
    s1 = s1tab('train')
    g = gt('train').with_columns(src=SRC)
    per = (s1.join(g.group_by('s1').agg(n=pl.len(), n2=(pl.col('src') == 2).sum(), n3=(pl.col('src') == 3).sum()),
                   on='s1', how='left').fill_null(0))
    h = np.bincount(np.minimum(per['n'].to_numpy(), KMAX), minlength=KMAX + 1).astype(np.float64)
    cap = {2: int(per['n2'].max()), 3: int(per['n3'].max())}
    const = dict(empty=float((per['n'] == 0).mean()), zero_s2=float((per['n2'] == 0).mean()),
                 zero_s3=float((per['n3'] == 0).mean()), s2_per_s1=float(per['n2'].mean()),
                 s3_per_s1=float(per['n3'].mean()))
    return h / h.sum(), cap, const


def shift(p, s1, fr=None, fr2=None, fr3=None):
    """logit shift for pairs whose S1 country is unseen in training; fr2 / fr3 override fr for S2 / S3 records."""
    if fr is None and fr2 is None and fr3 is None:
        return p
    cs = unseen(s1.rename({'s1': 'id'}))
    d2 = fr2 if fr2 is not None else (fr or 0.0)
    d3 = fr3 if fr3 is not None else (fr or 0.0)
    lg = (pl.col('p').clip(1e-7, 1 - 1e-7) / (1 - pl.col('p').clip(1e-7, 1 - 1e-7))).log()
    dl = pl.when(SRC == 2).then(pl.lit(d2)).otherwise(pl.lit(d3))
    p = p.join(s1.rename({'s1': 'id1'}), on='id1', how='left')
    return p.with_columns(p=pl.when(pl.col('country').is_in(cs)).then(1 / (1 + (-(lg + dl)).exp()))
                          .otherwise(pl.col('p'))).select('id1', 'id2', 'p')


def prep(p, cfg):
    """one-to-one + the q / floor / order of decide.pick_ef."""
    d = one2one(p.select('id1', 'id2', 'p'), cfg.get('margin', 0.0))
    d = d.with_columns(q=(pl.col('p') ** cfg['a']).clip(1e-6, 1 - 1e-6)).filter(pl.col('q') >= FLOOR)
    return d.sort(['id1', 'q'], descending=[False, True]).with_columns(k=pl.col('q').cum_count().over('id1'))


def pmf(d, s1):
    """Poisson-binomial PMF of K per S1 (rows aligned with s1), from the q of its prepped pairs."""
    idx = s1.select(pl.col('s1').alias('id1')).with_row_index('i')
    x = d.join(idx, on='id1', how='inner').select('i', 'k', 'q')
    P = np.zeros((s1.height, KMAX + 1))
    P[:, 0] = 1.0
    i, k, q = x['i'].to_numpy(), x['k'].to_numpy(), x['q'].to_numpy()
    for r in range(1, int(k.max()) + 1 if len(k) else 1):
        m = k == r
        ii, qq = i[m], q[m][:, None]
        old = P[ii]
        new = old * (1 - qq)
        new[:, 1:] += old[:, :-1] * qq
        new[:, -1] += old[:, -1] * qq[:, 0]
        P[ii] = new
    return P


def posterior(d, s1, prior, lam):
    """-> (id1, e0p, Sp): count-prior posterior P(K=0) and E[K] per S1."""
    P = pmf(d, s1)
    ctry = s1['country'].to_numpy()
    post = np.empty_like(P)
    for c in np.unique(ctry):
        m = ctry == c
        base = P[m].mean(0)
        w = np.where(base > 1e-12, (prior / np.maximum(base, 1e-12)) ** lam, 1.0)
        post[m] = P[m] * w[None, :]
    post /= post.sum(1, keepdims=True)
    return s1.select(pl.col('s1').alias('id1')).with_columns(
        e0p=pl.Series(post[:, 0]), Sp=pl.Series(post @ np.arange(KMAX + 1, dtype=np.float64)))


def pick(d, cfg, adj=None):
    """decide.pick_ef on prepped pairs; adj (id1, e0p, Sp) replaces prod(1-q) and sum(q)."""
    b2 = B * B
    d = d.with_columns(C=pl.col('q').cum_sum().over('id1'), S=pl.col('q').sum().over('id1'),
                       e0=(1 - pl.col('q')).log().sum().over('id1').exp())
    if adj is not None:
        d = d.join(adj, on='id1', how='left').with_columns(e0=pl.col('e0p'), S=pl.col('Sp')).drop('e0p', 'Sp')
    d = d.with_columns(ef=(1 + b2) * pl.col('C') / (b2 * (pl.col('S') + cfg['m']) + pl.col('k')))
    d = d.with_columns(efm=pl.col('ef').max().over('id1'))
    kb = d.filter(pl.col('ef') == pl.col('efm')).group_by('id1').agg(pl.col('k').min().alias('kb'))
    d = d.join(kb, on='id1').filter((pl.col('k') <= pl.col('kb')) & (pl.col('efm') > pl.col('e0')))
    return d.select('id1', 'id2', 'q')


def capped(sel, cap):
    sel = sel.with_columns(src=SRC).with_columns(r=pl.col('q').rank('ordinal', descending=True).over('id1', 'src'))
    return sel.filter(((pl.col('src') == 2) & (pl.col('r') <= cap[2])) | ((pl.col('src') == 3) & (pl.col('r') <= cap[3])))


def decide_variant(p, s1, cfg, caps=False, lam=None, prior=None, cap=None):
    assert cfg['method'] == 'ef', 'qfix supports the expected-F rule only'
    d = prep(p, cfg)
    sel = pick(d, cfg, posterior(d, s1, prior, lam) if lam else None)
    if caps:
        sel = capped(sel, cap)
    return sel.select('id1', 'id2')


def moments(sel, s1):
    x = sel.with_columns(src=SRC).group_by('id1').agg(n=pl.len(), n2=(pl.col('src') == 2).sum(), n3=(pl.col('src') == 3).sum())
    per = s1.rename({'s1': 'id1'}).join(x, on='id1', how='left').fill_null(0)
    return per.group_by('country').agg(
        empty=(pl.col('n') == 0).mean(), links=pl.col('n').mean(), s2=pl.col('n2').mean(), s3=pl.col('n3').mean(),
        zero_s2=(pl.col('n2') == 0).mean(), zero_s3=(pl.col('n3') == 0).mean(), n_s1=pl.len()).sort('country')


def cmd_tune(kv):
    tag = kv['tag']
    cfg = json.load(open(f'{W}/dec_{tag}.json'))
    prior, cap, const = train_stats()
    print('train prior P(K=k):', np.round(prior, 4).tolist(), '\ncaps', cap, 'constants', const, flush=True)
    s1 = s1tab('tsim').with_columns(fold=(pl.col('s1').hash(seed=7) % 5).cast(pl.Int8))
    g = gt('tsim')
    p = pl.read_parquet(f'{W}/oof_{tag}.parquet', columns=['id1', 'id2', 'p'])
    variants = [('base', {}), ('caps', dict(caps=True))]
    for lam in [float(x) for x in kv.get('lams', '0.5:1').replace(',', ':').split(':')]:
        variants += [(f'prior{lam:g}', dict(lam=lam)), (f'caps+prior{lam:g}', dict(caps=True, lam=lam))]
    res, base = {}, None
    for name, v in variants:
        sel = decide_variant(p, s1.select('s1', 'country'), cfg, prior=prior, cap=cap, **v)
        r = score(sel, s1.select('s1', 'country'), g)
        r['folds'] = [score(sel, s1.filter(pl.col('fold') == f).select('s1', 'country'), g, False)['ALL'] for f in range(5)]
        r['links'] = sel.height
        res[name] = r
        if base is None:
            base = r
            print(f'base reproduces dec_{tag}.json: {r["ALL"]:.6f} vs {cfg["oof"]["ALL"]:.6f}', flush=True)
        df = [a - b for a, b in zip(r['folds'], base['folds'])]
        print(f'{name:16s} ALL {r["ALL"]:.6f} ({r["ALL"] - base["ALL"]:+.6f})  US {r["US"]:.6f}  India {r["India"]:.6f}  '
              f'empty {r["pred_empty"]:.4f}  links {r["links"]}  folds+ {sum(x > 0 for x in df)}/5 '
              f'{" ".join(f"{x:+.6f}" for x in df)}', flush=True)
    json.dump(res, open(f'{W}/qfix_tune_{tag}.json', 'w'), indent=1)


def cmd_diag(kv):
    tag = kv['tag']
    cfg = json.load(open(f'{W}/dec_{tag}.json'))
    prior, cap, const = train_stats()
    print('train ground-truth constants', {k: round(v, 4) for k, v in const.items()}, flush=True)
    s1 = s1tab('test')
    cs = unseen(s1.rename({'s1': 'id'}))
    p = pl.read_parquet(f'{W}/pte_{tag}.parquet', columns=['id1', 'id2', 'p'])
    with pl.Config(tbl_rows=40, tbl_cols=12, tbl_width_chars=200, float_precision=4):
        for fr in [float(x) for x in kv.get('fr', '-3:-2.5:-2:-1.5:-1:0').replace(',', ':').split(':')]:  # ':' survives sbatch --export
            sel = decide_variant(shift(p, s1, fr=fr), s1, cfg, caps=kv.get('caps') == '1', prior=prior, cap=cap,
                                 lam=float(kv['prior']) if 'prior' in kv else None)
            m = moments(sel, s1)
            seen = m.filter(~pl.col('country').is_in(cs))
            tgt = {c: float((seen[c] * seen['n_s1']).sum() / seen['n_s1'].sum()) for c in ('empty', 'links', 's2', 's3')}
            un = m.filter(pl.col('country').is_in(cs))
            gap = {c: float(un[c][0]) - tgt[c] for c in tgt} if un.height else {}
            print(f'\n== {tag} unseen-country shift {fr:+.1f}  (seen-country target {tgt})\n{m}\n'
                  f'unseen minus seen: ' + '  '.join(f'{c} {v:+.4f}' for c, v in gap.items()), flush=True)


def cmd_apply(kv):
    tag, out = kv['tag'], kv['out']
    cfg = json.load(open(f'{W}/dec_{tag}.json'))
    prior, cap, _ = train_stats()
    caps = kv.get('caps') == '1'
    lam = float(kv['prior']) if 'prior' in kv else None
    f = {k: float(kv[k]) if k in kv else None for k in ('fr', 'fr2', 'fr3')}
    s1v = s1tab('tsim')
    selv = decide_variant(pl.read_parquet(f'{W}/oof_{tag}.parquet', columns=['id1', 'id2', 'p']), s1v, cfg,
                          caps=caps, lam=lam, prior=prior, cap=cap)
    oof = score(selv, s1v, gt('tsim'))
    print('tsim', json.dumps(oof), flush=True)
    s1 = s1tab('test')
    sel = decide_variant(shift(pl.read_parquet(f'{W}/pte_{tag}.parquet', columns=['id1', 'id2', 'p']), s1, **f),
                         s1, cfg, caps=caps, lam=lam, prior=prior, cap=cap)
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
    json.dump(dict(cfg, oof=oof, split='tsim', base=tag, caps=caps, prior=lam, cap=cap if caps else None, **f),
              open(f'{W}/dec_{out}.json', 'w'))


if __name__ == '__main__':
    c = sys.argv[1]
    kv = dict(a.split('=', 1) for a in sys.argv[2:])
    dict(tune=cmd_tune, diag=cmd_diag, apply=cmd_apply)[c](kv)
