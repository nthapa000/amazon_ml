"""K-fold GBDT baseline + tuner. Saves OOF and test predictions for blending.

python gbdt_cv.py train=train.csv test=test.csv target=price id=sample_id folds=folds.csv \
    model=lgb metric=smape tf=log feats=feat_train.csv:feat_test.csv tfidf=catalog_content trials=0 tag=lgb1

model   lgb | xgb | cat
task    auto | reg | clf
tf      log | log1p | none      (regression target transform; log for positive skewed targets)
obj     auto | l1 | l2 | huber  (auto: l1 for smape/mape/mae, l2 otherwise; in transformed space)
feats   extra feature files as trainfile:testfile pairs, comma separated, merged on id
tfidf   text column -> word+char TF-IDF -> SVD(svd) components (fit on train+test text)
te      categorical columns to K-fold target-encode inside each fold (no leakage)
trials  Optuna trials (0 = defaults). Tunes on the same folds, then refits.
params  params_<tag>.json from an earlier tuning run: reuse its best params (e.g. at a lower lr)
drop    columns to exclude
Outputs: oof_<tag>.npy, tst_<tag>.npy, imp_<tag>.csv, params_<tag>.json
"""
import json
import time
import numpy as np
import pandas as pd
from kv import args, cols, read, is_text
from metrics import M

a = args({'train': 'train.csv', 'test': 'test.csv', 'target': 'target', 'id': '', 'folds': 'folds.csv',
          'model': 'lgb', 'task': 'auto', 'metric': 'smape', 'tf': 'log', 'obj': 'auto', 'feats': '',
          'tfidf': '', 'svd': 64, 'te': '', 'trials': 0, 'drop': '', 'tag': '', 'seed': 42,
          'lr': 0.03, 'rounds': 20000, 'es': 300, 'maxcat': 1000, 'params': ''})
tag = a['tag'] or a['model']
T, I = a['target'], a['id']
tr, te = read(a['train']), read(a['test'])
for pair in cols(a['feats']):
    ftr, fte = pair.split(':')
    tr = tr.merge(read(ftr), on=I, how='left', suffixes=('', '_f'))
    te = te.merge(read(fte), on=I, how='left', suffixes=('', '_f'))
fo = read(a['folds'])
fold = tr[I].map(dict(zip(fo[fo.columns[0]], fo['fold']))).values if I else fo['fold'].values
assert not pd.isna(fold).any(), 'fold file does not cover all train rows'
fold = fold.astype(int)
K = fold.max() + 1

fn, mode, kind = M[a['metric']]
task = a['task'] if a['task'] != 'auto' else kind
y_raw = tr[T].values
if task == 'reg':
    y_raw = y_raw.astype(float)
    fwd = {'log': np.log, 'log1p': np.log1p, 'none': lambda v: v}[a['tf']]
    inv = {'log': np.exp, 'log1p': np.expm1, 'none': lambda v: v}[a['tf']]
    y = fwd(y_raw)
    ncls = 0
else:
    cl, y = np.unique(tr[T].astype(str), return_inverse=True)
    ncls = len(cl)

# ---------- features ----------
X = pd.concat([tr.drop(columns=[T]), te], keys=['tr', 'te'])
drop = set(cols(a['drop'])) | ({I} if I else set())
if a['tfidf']:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from scipy.sparse import hstack
    txt = X[a['tfidf']].fillna('').astype(str).str.lower()
    w = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200000, sublinear_tf=True).fit_transform(txt)
    c = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=5, max_features=200000, sublinear_tf=True).fit_transform(txt)
    Z = TruncatedSVD(a['svd'], random_state=a['seed']).fit_transform(hstack([w, c]).tocsr())
    for j in range(Z.shape[1]):
        X[f'svd{j}'] = Z[:, j]
cat = []
for c in X.columns:
    if c in drop:
        continue
    if is_text(X[c]) or str(X[c].dtype) == 'category':
        if X[c].nunique() <= a['maxcat']:
            X[c] = X[c].astype('category').cat.codes.astype('int32')
            X.loc[X[c] < 0, c] = -1
            cat.append(c)
        else:
            drop.add(c)
bad = [c for c in drop if c in X]
if bad:
    print('dropped (id / text / high-card):', bad[:15], '...' if len(bad) > 15 else '')
X = X.drop(columns=bad)
Xtr, Xte = X.loc['tr'].reset_index(drop=True), X.loc['te'].reset_index(drop=True)
te_cols = cols(a['te'])
print(f'task={task} metric={a["metric"]} model={a["model"]} features={Xtr.shape[1]} folds={K}')


def add_te(xt, xv, yt, cs, k=20):
    """smoothed mean target per category, fit on training part only."""
    xt, xv = xt.copy(), xv.copy()
    g = yt.mean()
    for c in cs:
        d = pd.DataFrame({'c': xt[c].values, 'y': yt})
        s = d.groupby('c')['y'].agg(['mean', 'count'])
        m = (s['mean'] * s['count'] + g * k) / (s['count'] + k)
        xt[c + '_te'] = xt[c].map(m).fillna(g).values
        xv[c + '_te'] = xv[c].map(m).fillna(g).values
    return xt, xv


def obj_name():
    o = a['obj']
    if o == 'auto':
        o = 'l1' if a['metric'] in ('smape', 'mape', 'mape_score', 'mae') else 'l2'
    return o


def fit_one(xt, yt, xv, yv, p):
    m = a['model']
    o = obj_name()
    if m == 'lgb':
        import lightgbm as lgb
        q = dict(learning_rate=a['lr'], num_leaves=63, min_data_in_leaf=20, feature_fraction=.8,
                 bagging_fraction=.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=a['seed'])
        if task == 'reg':
            q['objective'] = {'l1': 'l1', 'l2': 'l2', 'huber': 'huber'}[o]
        else:
            q['objective'] = 'binary' if ncls == 2 else 'multiclass'
            if ncls > 2:
                q['num_class'] = ncls
        q.update(p)
        dt = lgb.Dataset(xt, yt, categorical_feature=[c for c in cat if c in xt] or 'auto')
        dv = lgb.Dataset(xv, yv, reference=dt)
        b = lgb.train(q, dt, a['rounds'], valid_sets=[dv],
                      callbacks=[lgb.early_stopping(a['es'], verbose=False)])
        pr = lambda z: b.predict(z, num_iteration=b.best_iteration)
        imp = pd.Series(b.feature_importance('gain'), index=xt.columns)
        return pr, imp, b.best_iteration
    if m == 'xgb':
        import xgboost as xgb
        q = dict(eta=a['lr'], max_depth=8, subsample=.8, colsample_bytree=.8, min_child_weight=5,
                 reg_lambda=1.0, tree_method='hist', seed=a['seed'])
        if task == 'reg':
            q['objective'] = {'l1': 'reg:absoluteerror', 'l2': 'reg:squarederror', 'huber': 'reg:pseudohubererror'}[o]
        else:
            q['objective'] = 'binary:logistic' if ncls == 2 else 'multi:softprob'
            if ncls > 2:
                q['num_class'] = ncls
        q.update(p)
        dt, dv = xgb.DMatrix(xt, yt), xgb.DMatrix(xv, yv)
        b = xgb.train(q, dt, a['rounds'], evals=[(dv, 'v')], early_stopping_rounds=a['es'], verbose_eval=False)
        pr = lambda z: b.predict(xgb.DMatrix(z), iteration_range=(0, b.best_iteration + 1))
        imp = pd.Series(b.get_score(importance_type='gain')).reindex(xt.columns).fillna(0)
        return pr, imp, b.best_iteration
    from catboost import CatBoostRegressor, CatBoostClassifier
    q = dict(learning_rate=a['lr'] * 2, depth=8, iterations=a['rounds'], od_type='Iter', od_wait=a['es'],
             random_seed=a['seed'], verbose=False)
    q.update(p)
    ci = [xt.columns.get_loc(c) for c in cat if c in xt]
    if task == 'reg':
        q['loss_function'] = {'l1': 'MAE', 'l2': 'RMSE', 'huber': 'Huber:delta=1.0'}[o]
        b = CatBoostRegressor(**q)
    else:
        b = CatBoostClassifier(**q)
    b.fit(xt, yt, eval_set=(xv, yv), cat_features=ci)
    if task == 'reg':
        pr = b.predict
    else:
        pr = (lambda z: b.predict_proba(z)[:, 1]) if ncls == 2 else b.predict_proba
    return pr, pd.Series(b.get_feature_importance(), index=xt.columns), b.get_best_iteration()


def to_score(yv_raw, pv):
    if task == 'reg':
        return fn(yv_raw, inv(pv))
    if a['metric'] == 'auc':
        return fn(yv_raw, pv)
    lab = (pv > .5).astype(int) if ncls == 2 else pv.argmax(1)
    return fn(yv_raw, lab)


def run(p, save=False):
    shp = (len(Xtr),) if (task == 'reg' or ncls == 2) else (len(Xtr), ncls)
    oof = np.zeros(shp)
    tst = np.zeros((len(Xte),) if len(shp) == 1 else (len(Xte), ncls))
    imps, its = [], []
    for f in range(K):
        t, v = fold != f, fold == f
        xt, xv, xs = Xtr[t], Xtr[v], Xte
        if te_cols:
            xt, xv = add_te(xt, xv, y[t], te_cols)
            _, xs = add_te(Xtr[t], Xte, y[t], te_cols)
        pr, imp, it = fit_one(xt, y[t], xv, y[v], p)
        oof[v] = pr(xv)
        tst += pr(xs) / K
        imps.append(imp); its.append(it)
        if save:
            yv = y_raw[v] if task == 'reg' else y[v]
            print(f'  fold {f}: {a["metric"]}={to_score(yv, oof[v]):.4f}  best_iter={it}')
    yall = y_raw if task == 'reg' else y
    return to_score(yall, oof), oof, tst, pd.concat(imps, axis=1).mean(axis=1), its


best = json.load(open(a['params']))['best'] if a['params'] else {}
if best:
    print('loaded params', best)
if a['trials'] > 0 and not best:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def space(tr_):
        if a['model'] == 'lgb':
            return dict(num_leaves=tr_.suggest_int('num_leaves', 15, 255, log=True),
                        min_data_in_leaf=tr_.suggest_int('min_data_in_leaf', 5, 200, log=True),
                        feature_fraction=tr_.suggest_float('feature_fraction', .4, 1.),
                        bagging_fraction=tr_.suggest_float('bagging_fraction', .5, 1.),
                        lambda_l1=tr_.suggest_float('lambda_l1', 1e-3, 10, log=True),
                        lambda_l2=tr_.suggest_float('lambda_l2', 1e-3, 10, log=True))
        if a['model'] == 'xgb':
            return dict(max_depth=tr_.suggest_int('max_depth', 3, 12),
                        min_child_weight=tr_.suggest_float('min_child_weight', 1, 50, log=True),
                        subsample=tr_.suggest_float('subsample', .5, 1.),
                        colsample_bytree=tr_.suggest_float('colsample_bytree', .4, 1.),
                        reg_alpha=tr_.suggest_float('reg_alpha', 1e-3, 10, log=True),
                        reg_lambda=tr_.suggest_float('reg_lambda', 1e-3, 10, log=True))
        return dict(depth=tr_.suggest_int('depth', 4, 10),
                    l2_leaf_reg=tr_.suggest_float('l2_leaf_reg', 1, 30, log=True),
                    random_strength=tr_.suggest_float('random_strength', 1e-2, 10, log=True))

    st = optuna.create_study(direction='minimize' if mode == 'min' else 'maximize',
                             sampler=optuna.samplers.TPESampler(seed=a['seed']))
    st.optimize(lambda t_: run(space(t_))[0], n_trials=a['trials'])
    best = st.best_params
    print('optuna best', round(st.best_value, 4), best)

t0 = time.time()
sc, oof, tst, imp, its = run(best, save=True)
print(f'OOF {a["metric"]} = {sc:.4f}   ({time.time() - t0:.0f}s)')
if task == 'reg':
    oof, tst = inv(oof), inv(tst)          # log space average -> geometric mean of fold models
np.save(f'oof_{tag}.npy', oof)
np.save(f'tst_{tag}.npy', tst)
imp.sort_values(ascending=False).to_csv(f'imp_{tag}.csv', header=['gain'])
json.dump({'args': a, 'best': best, 'oof': sc, 'iters': [int(i) for i in its]},
          open(f'params_{tag}.json', 'w'), indent=1, default=str)
print('top features:', ', '.join(imp.sort_values(ascending=False).head(12).index))
print(f'saved oof_{tag}.npy tst_{tag}.npy imp_{tag}.csv params_{tag}.json')
