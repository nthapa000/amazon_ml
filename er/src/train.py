"""LightGBM matcher with 5-fold GroupKFold by S1 id (hash) -> OOF on train pairs + averaged test preds.

python train.py tr=f_cand1_train te=f_cand1_test tag=s1 frac=0.35 [extra=ctx_train:ctx_test]
  [models=<tag> alias=new:old  (no training: re-score test with the saved fold models of <tag>)]
Writes W/oof_{tag}.parquet (id1,id2,y,p) and W/pte_{tag}.parquet (id1,id2,p).
"""
import os
import sys
import json
import time

import numpy as np
import polars as pl
import lightgbm as lgb

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')
NF = 5
DROP = {'id1', 'id2', 'y', 'fold', 'p', 'kbits'}

P = dict(objective='binary', learning_rate=0.08, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.7,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255, verbose=-1,
         num_threads=int(os.environ.get('SLURM_CPUS_PER_TASK', 32)))


sys.path.insert(0, os.path.dirname(__file__))
from block import enc, FAM, gt_path  # noqa: E402


def fold_of(ids):
    return (ids.hash(seed=7) % NF).cast(pl.Int8)


def load(name, extra=None, keep=None):
    """keep: optional DataFrame (id1, id2) -> cascade: only these pairs are read (per part, lazily)."""
    p = f'{W}/{name}/*.parquet' if os.path.isdir(f'{W}/{name}') else f'{W}/{name}.parquet'
    import glob
    if keep is None:
        d = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(p))], how='vertical_relaxed')  # int widths may differ
    else:
        kl = keep.lazy()
        d = pl.concat([pl.scan_parquet(f).join(kl, on=['id1', 'id2'], how='semi').collect()
                       for f in sorted(glob.glob(p))], how='vertical_relaxed')
    for e in (extra.split('+') if extra else []):
        if keep is None:
            x = pl.read_parquet(f'{W}/{e}.parquet')
        else:
            x = pl.scan_parquet(f'{W}/{e}.parquet').join(d.select('id1', 'id2').lazy(), on=['id1', 'id2'],
                                                         how='semi').collect()
        d = d.join(x, on=['id1', 'id2'], how='left')
    if 'kbits' in d.columns:
        d = d.with_columns([((pl.col('kbits') & v[0]) > 0).cast(pl.Int8).alias('k_' + f) for f, v in FAM.items()])
    return d


def predict_only(kv, ex, keep, t0):
    """models=<tag>: reuse the saved fold models of an earlier run with the same training data and only re-score the test
    pairs, e.g. after a test-side feature was patched (ce.py merge unseen=1). alias=new:old[+...] renames a test column to
    the name the models were trained with. OOF = the earlier run's OOF (same models)."""
    src = kv['models']
    models = [lgb.Booster(model_file=f'{W}/lgb_{src}_{k}.txt') for k in range(NF)]   # saved at the best iteration
    feats = models[0].feature_name()
    pl.read_parquet(f'{W}/oof_{src}.parquet').write_parquet(f'{W}/oof_{kv["tag"]}.parquet')
    te = load(kv['te'], ex[1], keep[1])
    for a in kv['alias'].split('+') if kv.get('alias') else []:
        new, old = a.split(':')
        te = te.rename({new: old})
    pte = np.zeros(te.height, dtype=np.float32)
    step = 5_000_000
    for i in range(0, te.height, step):
        Xs = te.slice(i, step).select(feats).to_numpy().astype(np.float32)
        pte[i:i + Xs.shape[0]] = np.mean([m.predict(Xs) for m in models], 0)
    te.select('id1', 'id2').with_columns(p=pte).write_parquet(f'{W}/pte_{kv["tag"]}.parquet')
    print(f'models of {src}: test re-scored {te.height} {time.time() - t0:.0f}s', flush=True)


def main(kv):
    t0 = time.time()
    ex = kv.get('extra', '').split(':') if kv.get('extra') else [None, None]
    keep = [None, None]
    if kv.get('sub'):                                # cascade: sub=<oof_tag>:<pte_tag>:<thr>
        a, b, thr = kv['sub'].split(':')
        keep = [pl.scan_parquet(f'{W}/{f}.parquet').filter(pl.col('p') >= float(thr)).select('id1', 'id2').collect()
                for f in (a, b)]
        print('cascade keep', keep[0].height, keep[1].height, flush=True)
    if kv.get('models'):
        return predict_only(kv, ex, keep, t0)
    tr = load(kv['tr'], ex[0], keep[0])
    gt = pl.read_parquet(gt_path(kv.get('split', 'train'))).select(enc('s1').alias('id1'), enc('id2').alias('id2'),
                                                          y=pl.lit(1, pl.Int8))
    tr = tr.join(gt, on=['id1', 'id2'], how='left').with_columns(pl.col('y').fill_null(0), fold=fold_of(pl.col('id1')))
    feats = [c for c in tr.columns if c not in DROP]
    print('train pairs', tr.shape, 'pos', tr['y'].sum(), 'feats', len(feats), flush=True)
    frac = float(kv.get('frac', 0.35))
    rounds = int(kv.get('rounds', 1500))
    seed = int(kv.get('seed', 0))
    params = dict(P, **json.loads(kv.get('params', '{}')))
    if kv.get('pfile'):                              # tuned params from tune.py (file avoids commas in sbatch)
        params.update(json.load(open(f'{W}/{kv["pfile"]}')))
    params.update(seed=seed, bagging_seed=seed + 1, feature_fraction_seed=seed + 2)
    if 'lr' in kv:
        params['learning_rate'] = float(kv['lr'])
    ids = tr.select('id1', 'id2', 'y')
    fold = tr['fold'].to_numpy()
    # S1-level subsample for training (keeps each S1's full candidate list together)
    samp = (tr['id1'].hash(seed=11 + seed) % 1000 < int(frac * 1000)).to_numpy()   # seed changes the S1 subsample
    X = tr.select(feats).to_numpy().astype(np.float32)
    y = tr['y'].to_numpy()
    del tr                                           # keep only one copy of the big matrix
    oof = np.zeros(X.shape[0], dtype=np.float32)
    imp = np.zeros(len(feats))
    models = []
    for k in range(NF):
        trm, vam = (fold != k) & samp, fold == k
        vs = vam & samp                              # early-stop on a slice of the held-out fold
        if kv.get('algo') == 'xgb':                   # second model family for the blend
            import xgboost as xgb
            xp = dict(objective='binary:logistic', eval_metric='logloss', eta=params['learning_rate'],
                      max_depth=int(kv.get('depth', 10)), min_child_weight=20, subsample=0.8, colsample_bytree=0.7,
                      reg_lambda=1.0, tree_method='hist', max_bin=255, nthread=P['num_threads'], seed=seed)
            dtr = xgb.DMatrix(X[trm], y[trm], feature_names=feats)
            dva = xgb.DMatrix(X[vs], y[vs], feature_names=feats)
            m = xgb.train(xp, dtr, rounds, evals=[(dva, 'va')], early_stopping_rounds=50, verbose_eval=200)
            m.best_iteration = m.best_iteration + 1   # number of trees to use
            m.predict_x = lambda Z, m=m: m.predict(xgb.DMatrix(Z, feature_names=feats),
                                                   iteration_range=(0, m.best_iteration))
            g = m.get_score(importance_type='total_gain')
            imp += np.array([g.get(f, 0.0) for f in feats])
            m.save_model(f'{W}/xgb_{kv["tag"]}_{k}.json')
        else:
            dtr = lgb.Dataset(X[trm], y[trm], feature_name=feats, free_raw_data=True)
            dva = lgb.Dataset(X[vs], y[vs], reference=dtr)
            m = lgb.train(params, dtr, rounds, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)])
            m.predict_x = lambda Z, m=m: m.predict(Z, num_iteration=m.best_iteration)
            imp += m.feature_importance('gain')
            m.save_model(f'{W}/lgb_{kv["tag"]}_{k}.txt')
        oof[vam] = m.predict_x(X[vam])
        models.append(m)
        del dtr, dva
        print(f'fold {k} it={m.best_iteration} {time.time() - t0:.0f}s', flush=True)
    ids.with_columns(p=oof).write_parquet(f'{W}/oof_{kv["tag"]}.parquet')
    del X
    if kv.get('te'):
        te = load(kv['te'], ex[1], keep[1])
        pte = np.zeros(te.height, dtype=np.float32)
        step = 5_000_000
        for i in range(0, te.height, step):          # score test in slices
            Xs = te.slice(i, step).select(feats).to_numpy().astype(np.float32)
            pte[i:i + Xs.shape[0]] = np.mean([m.predict_x(Xs) for m in models], 0)
        te.select('id1', 'id2').with_columns(p=pte).write_parquet(f'{W}/pte_{kv["tag"]}.parquet')
        print(f'test scored {te.height} {time.time() - t0:.0f}s', flush=True)
    for f, g in sorted(zip(feats, imp), key=lambda x: -x[1])[:40]:
        print(f'  {f:14s} {g / imp.sum():.4f}')


if __name__ == '__main__':
    main(dict(a.split('=', 1) for a in sys.argv[1:]))
