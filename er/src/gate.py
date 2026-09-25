"""Entity-level gate: learned P(S1 has no match) for the empty-list decision (targets predicted-empty and
singleton-false-merge loss). Features come from the stage-2 scores of the S1's candidates (after the version's
one-to-one assignment) plus name / address commonness. 5-fold by the same S1 hash as train.py -> OOF on tsim,
fold-averaged on test. Then g (multiplier on q0) is tuned on OOF with the version's decision rule.

python gate.py tag=c2 out=c2g        -> W/gate_{tag}_train.parquet, W/gate_{tag}_test.parquet (id1, q0),
                                        W/dec_{out}.json (+ oof/pte symlinks for release)
"""
import json
import os
import sys

import numpy as np
import polars as pl
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__))
from block import enc, gt_path  # noqa: E402
from decide import one2one, select, score  # noqa: E402
from train import fold_of  # noqa: E402

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')


def ent_feats(p, split, margin):
    """p: (id1, id2, p) stage-2 scores -> one row per S1 of the split."""
    s1 = pl.read_parquet(f'{W}/n_{split}_s1.parquet', columns=['id', 'core', 'ad'])
    s1 = s1.with_columns(id1=enc('id')).drop('id')
    s1 = s1.with_columns(n_c1=pl.len().over('core').cast(pl.Int32), n_ca1=pl.len().over('ad').cast(pl.Int32),
                         ad_empty=(pl.col('ad') == '').cast(pl.Int8),
                         nt=pl.col('core').str.split(' ').list.len().cast(pl.Int16))
    a = p.group_by('id1').agg(pl.len().alias('n_all'), pl.col('p').max().alias('pmax_all'),
                              pl.col('p').sum().alias('psum_all'))
    o = one2one(p, margin)
    q = pl.col('p')
    b = o.group_by('id1').agg(
        pl.len().alias('n'), q.max().alias('pmax'), q.top_k(2).min().alias('p2'), q.sum().alias('psum'),
        (q > 0.9).sum().alias('n90'), (q > 0.5).sum().alias('n50'), (q > 0.2).sum().alias('n20'),
        (q > 0.05).sum().alias('n05'), (1 - q.clip(0, 1 - 1e-6)).log().sum().alias('lprod'),
        q.filter(q < 0.9).max().alias('pmax_lt90'))
    f = s1.join(a, on='id1', how='left').join(b, on='id1', how='left')
    return f.with_columns(lost=pl.col('n_all').fill_null(0) - pl.col('n').fill_null(0)).drop('core', 'ad')


def main(kv):
    tag, out = kv['tag'], kv.get('out', kv['tag'] + 'g')
    cfg = json.load(open(f'{W}/dec_{tag}.json'))
    base = {k: v for k, v in cfg.items() if k not in ('oof', 'split')}
    o = pl.read_parquet(f'{W}/oof_{tag}.parquet', columns=['id1', 'id2', 'p'])
    te = pl.read_parquet(f'{W}/pte_{tag}.parquet', columns=['id1', 'id2', 'p'])
    ftr, fte = ent_feats(o, 'tsim', cfg.get('margin', 0.0)), ent_feats(te, 'test', cfg.get('margin', 0.0))
    gt = pl.read_parquet(gt_path('tsim')).select(enc('s1').alias('id1'), enc('id2').alias('id2'))
    ftr = ftr.join(gt.group_by('id1').len('nt_true'), on='id1', how='left').with_columns(
        y=(pl.col('nt_true').fill_null(0) == 0).cast(pl.Int8), fold=fold_of(pl.col('id1')))
    feats = [c for c in ftr.columns if c not in ('id1', 'y', 'fold', 'nt_true')]
    X, y, fo = ftr.select(feats).to_numpy().astype(np.float32), ftr['y'].to_numpy(), ftr['fold'].to_numpy()
    Xt = fte.select(feats).to_numpy().astype(np.float32)
    P = dict(objective='binary', learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=int(os.environ.get('SLURM_CPUS_PER_TASK', 16)))
    oof, pt = np.zeros(len(y), np.float32), np.zeros(Xt.shape[0], np.float32)
    for k in range(5):
        tr, va = fo != k, fo == k
        m = lgb.train(P, lgb.Dataset(X[tr], y[tr], feature_name=feats), 3000,
                      valid_sets=[lgb.Dataset(X[va], y[va])], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va] = m.predict(X[va], num_iteration=m.best_iteration)
        pt += m.predict(Xt, num_iteration=m.best_iteration) / 5
        print(f'fold {k} it={m.best_iteration}', flush=True)
    from sklearn.metrics import roc_auc_score
    print('gate AUC (no-match) OOF', roc_auc_score(y, oof), ' singleton rate', y.mean(), flush=True)
    ftr.select('id1').with_columns(q0=pl.Series(oof)).write_parquet(f'{W}/gate_{tag}_train.parquet')
    fte.select('id1').with_columns(q0=pl.Series(pt)).write_parquet(f'{W}/gate_{tag}_test.parquet')
    gate = ftr.select('id1').with_columns(q0=pl.Series(oof))
    s1 = pl.read_parquet(f'{W}/n_tsim_s1.parquet', columns=['id', 'country']).select(enc('id').alias('s1'), 'country')
    gts = gt.rename({'id1': 's1'})
    best = (score(select(o, base), s1, gts, False)['ALL'], None)
    print(f'no gate: {best[0]:.6f}', flush=True)
    for g in (0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0):
        c = dict(base, gate=f'gate_{tag}', g=g)
        f = score(select(o, c, gate), s1, gts, False)['ALL']
        print(f'gate g={g}: {f:.6f}', flush=True)
        if f > best[0]:
            best = (f, g)
    if best[1] is None:
        print('gate does not help -> keep', tag)
        return
    c = dict(base, gate=f'gate_{tag}', g=best[1])
    sc = score(select(o, c, gate), s1, gts)
    json.dump(dict(c, oof=sc, split='tsim'), open(f'{W}/dec_{out}.json', 'w'))
    for kind in ('oof', 'pte'):
        dst = f'{W}/{kind}_{out}.parquet'
        if not os.path.exists(dst):
            os.symlink(f'{W}/{kind}_{tag}.parquet', dst)
    print('BEST', c, json.dumps(sc), flush=True)


if __name__ == '__main__':
    main(dict(a.split('=', 1) for a in sys.argv[1:]))
