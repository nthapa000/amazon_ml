"""Hour-0 EDA that prints findings AND the lever each finding points to.

python eda.py train=train.csv test=test.csv target=price id=sample_id
"""
import re
import numpy as np
import pandas as pd
from kv import args, read, is_text

a = args({'train': 'train.csv', 'test': '', 'target': 'target', 'id': '', 'n': 20})
tr = read(a['train'])
te = read(a['test']) if a['test'] else None
T, I = a['target'], a['id']
tips = []


def hr(t):
    print('\n' + '=' * 8 + ' ' + t + ' ' + '=' * 8)


hr('shape and memory')
print('train', tr.shape, f"{tr.memory_usage(deep=True).sum() / 2**20:.1f} MB")
if te is not None:
    print('test ', te.shape, f"{te.memory_usage(deep=True).sum() / 2**20:.1f} MB")
    miss_cols = set(tr.columns) - set(te.columns) - {T}
    if miss_cols:
        print('columns in train but not test:', sorted(miss_cols))
        tips.append(f'Train-only columns {sorted(miss_cols)} cannot be features (possible leakage).')
print(tr.dtypes.value_counts().to_string())

hr('target')
y = tr[T]
kind = 'clf' if (is_text(y) or y.nunique() <= 20) else 'reg'
print('task guess:', kind, '| unique:', y.nunique(), '| missing:', int(y.isna().sum()))
if kind == 'reg':
    q = y.quantile([0, .01, .25, .5, .75, .99, 1]).round(4)
    print(q.to_string())
    sk = float(y.skew())
    print('skew', round(sk, 2), '| min', y.min(), '| share<=0', round(float((y <= 0).mean()), 4))
    if sk > 1 and y.min() > 0:
        tips.append('Right-skewed positive target: model log(y), use L1/Huber, blend in log space (references/metrics.md).')
    top = y.value_counts(normalize=True).head(10)
    print('most common values (share):\n' + top.round(4).to_string())
    if top.sum() > 0.2:
        tips.append('Target repeats exact values often: try snap-to-grid post-processing (blend.py snap=1).')
    if y.quantile(.99) / max(y.quantile(.5), 1e-9) > 20:
        tips.append('Extreme upper tail: clip log target for training (e.g. 99.9th pct) and check worst-OOF rows for label errors.')
else:
    vc = y.value_counts(normalize=True)
    print(vc.head(20).round(4).to_string())
    if vc.min() < 0.1:
        tips.append('Imbalanced classes: stratified folds, scale_pos_weight / class weights, tune the decision threshold on OOF (metrics.tune_thr).')

hr('missing values (top)')
m = tr.isna().mean().sort_values(ascending=False)
m = m[m > 0]
print(m.head(a['n']).round(4).to_string() if len(m) else 'none')
if len(m):
    tips.append('Missing values: GBDTs handle NaN natively; add is_missing flags where missingness may carry signal; never drop TEST rows.')

hr('column profile')
rows = []
n = len(tr)
for c in tr.columns:
    if c == T:
        continue
    s = tr[c]
    nu = s.nunique(dropna=True)
    info = {'col': c, 'dtype': str(s.dtype), 'nunique': nu, 'miss': round(float(s.isna().mean()), 3)}
    if is_text(s):
        ln = s.dropna().astype(str).str.len()
        info['avg_len'] = round(float(ln.mean()), 1) if len(ln) else 0
    rows.append(info)
    idish = is_text(s) or pd.api.types.is_integer_dtype(s)
    if idish and nu >= 0.98 * n and c != I and info.get('avg_len', 0) < 40:
        tips.append(f'{c!r} is (almost) unique per row (ID-like, e.g. phone / serial no.): drop from features, keep for joins.')
    if nu <= 1:
        tips.append(f'{c!r} is constant: drop.')
    if is_text(s) and 50 < nu < 0.98 * n and info.get('avg_len', 0) < 40:
        tips.append(f'{c!r} is high-cardinality categorical: K-fold target encoding (gbdt_cv.py te=) or native categorical, not one-hot.')
    if is_text(s) and info.get('avg_len', 0) >= 40:
        tips.append(f'{c!r} is free text: parse numbers/units (parse_catalog.py), TF-IDF SVD (gbdt_cv.py tfidf=), embeddings + kNN (knn_feats.py).')
    if is_text(s) and nu == 2:
        tips.append(f'{c!r} is binary text: map to 0/1.')
print(pd.DataFrame(rows).to_string(index=False))

hr('redundant numeric columns (|r| > 0.995)')
num = tr.select_dtypes('number').drop(columns=[x for x in [T, I] if x in tr], errors='ignore')
if num.shape[1] > 1:
    cm = num.sample(min(n, 50000), random_state=0).corr().abs()
    up = cm.where(np.triu(np.ones(cm.shape), 1).astype(bool)).stack()
    red = up[up > 0.995]
    print(red.round(4).to_string() if len(red) else 'none')
    for (c1, c2), v in red.items():
        tips.append(f'{c1!r} and {c2!r} carry the same information (r={v:.3f}): keep one.')
dup = tr.T.duplicated()
if dup.any():
    tips.append(f'Exact duplicate columns: {list(tr.columns[dup])}')

hr('duplicates and train/test overlap')
feats = [c for c in tr.columns if c not in (T, I)]
print('duplicate train rows (features only):', int(tr.duplicated(subset=feats).sum()))
txt = [c for c in feats if is_text(tr[c]) and tr[c].astype(str).str.len().mean() >= 40]
if te is not None and txt:
    c = txt[0]
    norm = lambda s: s.astype(str).str.lower().map(lambda x: re.sub(r'\W+', ' ', x).strip())
    st = set(norm(tr[c]))
    ov = norm(te[c]).isin(st).mean()
    print(f'test rows whose {c!r} text exactly appears in train: {ov:.3%}')
    if ov > 0.02:
        tips.append('Near-duplicate listings across train/test: kNN neighbour-target features are likely strong; use group folds (make_folds.py group=).')
    same = tr.duplicated(subset=[c], keep=False).mean()
    if same > 0.02:
        tips.append(f'{same:.1%} of train rows share identical {c!r}: group folds by it to keep CV honest.')

hr('suggested next steps')
for i, t in enumerate(dict.fromkeys(tips), 1):
    print(f'{i}. {t}')
if not tips:
    print('No red flags. Build the baseline (gbdt_cv.py) and iterate.')
