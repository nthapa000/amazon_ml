"""Adversarial validation: can a model tell train rows from test rows?

python adv_val.py train=train.csv test=test.csv target=price id=sample_id drop=catalog_content,image_link
AUC ~0.5  -> train and test look alike; random/stratified folds are fine.
AUC >0.7  -> drift. Top features show what differs. Consider dropping them, or validating on the
             most test-like train rows (saved as advw.csv: id, p_test).
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from kv import args, cols, read, is_text

a = args({'train': 'train.csv', 'test': 'test.csv', 'target': 'target', 'id': 'sample_id',
          'drop': '', 'feats': '', 'out': 'advw.csv'})
tr, te = read(a['train']), read(a['test'])
for pair in cols(a['feats']):
    ftr, fte = pair.split(':')
    tr = tr.merge(read(ftr), on=a['id'], how='left'); te = te.merge(read(fte), on=a['id'], how='left')
X = pd.concat([tr.drop(columns=[a['target']]), te])
X = X.drop(columns=[c for c in cols(a['drop']) + [a['id']] if c in X])
for c in X.columns:
    if is_text(X[c]):
        X[c] = X[c].astype('category').cat.codes if X[c].nunique() < 1000 else X[c].astype(str).str.len()
z = np.r_[np.zeros(len(tr)), np.ones(len(te))]
p = np.zeros(len(z))
imp = pd.Series(0.0, index=X.columns)
for t, v in StratifiedKFold(5, shuffle=True, random_state=0).split(X, z):
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=.05, num_leaves=31, verbose=-1)
    m.fit(X.iloc[t], z[t])
    p[v] = m.predict_proba(X.iloc[v])[:, 1]
    imp += pd.Series(m.booster_.feature_importance('gain'), index=X.columns)
auc = roc_auc_score(z, p)
print(f'adversarial AUC = {auc:.3f}  ({"no meaningful drift" if auc < .6 else "DRIFT: inspect features below"})')
print(imp.sort_values(ascending=False).head(15).round(0).to_string())
pd.DataFrame({a['id']: tr[a['id']].values, 'p_test': p[:len(tr)]}).to_csv(a['out'], index=False)
print('wrote', a['out'], '(train rows with high p_test are the most test-like)')
