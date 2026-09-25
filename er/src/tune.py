"""Optuna search for stage-2 LightGBM parameters on the cascade subset (playbook 4G).

Objective = validation logloss on fold 0 (train on folds 1-4 of a sampled set of S1), so no F0.5 / decision
tuning leaks into the parameters. Best params go to W/params_{out}.json for train.py params=...

python tune.py tr=f_cand_tsim_kx extra=<train extras> sub=oof_w1:0.001 trials=30 frac=0.4 out=c
"""
import json
import os
import sys
import time

import numpy as np
import optuna
import polars as pl
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__))
from train import load, fold_of, DROP, P  # noqa: E402
from block import enc, gt_path  # noqa: E402

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')


def main(kv):
    t0 = time.time()
    f, thr = kv['sub'].split(':')
    keep = pl.scan_parquet(f'{W}/{f}.parquet').filter(pl.col('p') >= float(thr)).select('id1', 'id2').collect()
    keep = keep.filter(pl.col('id1').hash(seed=31) % 1000 < int(float(kv.get('frac', 0.4)) * 1000))
    d = load(kv['tr'], kv.get('extra'), keep)
    gt = pl.read_parquet(gt_path(kv.get('split', 'tsim'))).select(enc('s1').alias('id1'), enc('id2').alias('id2'),
                                                                   y=pl.lit(1, pl.Int8))
    d = d.join(gt, on=['id1', 'id2'], how='left').with_columns(pl.col('y').fill_null(0), fold=fold_of(pl.col('id1')))
    feats = [c for c in d.columns if c not in DROP]
    X, y, fo = d.select(feats).to_numpy().astype(np.float32), d['y'].to_numpy(), d['fold'].to_numpy()
    tr, va = fo != 0, fo == 0
    dtr = lgb.Dataset(X[tr], y[tr], feature_name=feats, free_raw_data=False)
    dva = lgb.Dataset(X[va], y[va], reference=dtr, free_raw_data=False)
    print('rows', len(y), 'pos', int(y.sum()), 'feats', len(feats), f'{time.time() - t0:.0f}s', flush=True)

    def obj(trial):
        p = dict(P, learning_rate=0.08,
                 num_leaves=trial.suggest_int('num_leaves', 31, 511, log=True),
                 min_data_in_leaf=trial.suggest_int('min_data_in_leaf', 20, 2000, log=True),
                 feature_fraction=trial.suggest_float('feature_fraction', 0.3, 1.0),
                 bagging_fraction=trial.suggest_float('bagging_fraction', 0.5, 1.0),
                 lambda_l2=trial.suggest_float('lambda_l2', 1e-3, 30, log=True),
                 min_gain_to_split=trial.suggest_float('min_gain_to_split', 0.0, 1.0))
        m = lgb.train(p, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
        trial.set_user_attr('it', m.best_iteration)
        return m.best_score['valid_0']['binary_logloss']

    st = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=0))
    st.enqueue_trial(dict(num_leaves=255, min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.8,
                          lambda_l2=1.0, min_gain_to_split=0.0))           # current default as trial 0
    st.optimize(obj, n_trials=int(kv.get('trials', 30)))
    for t in sorted(st.trials, key=lambda t: t.value)[:5]:
        print(f'{t.value:.6f} it={t.user_attrs.get("it")} {t.params}', flush=True)
    print('default logloss', st.trials[0].value, 'best', st.best_value, f'{time.time() - t0:.0f}s', flush=True)
    json.dump(st.best_params, open(f'{W}/params_{kv.get("out", "c")}.json', 'w'))


if __name__ == '__main__':
    main(dict(a.split('=', 1) for a in sys.argv[1:]))
