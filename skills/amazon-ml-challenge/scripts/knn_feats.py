"""Leak-free kNN neighbour-target features (the 'retrieval regressor' trick of 2025 top teams).

For each train row, neighbours come only from OTHER folds; for test rows, from all of train.
Input vectors: emb_tr/emb_te .npy files (any encoder: SigLIP, CLIP, sentence-transformer)
               OR a text column (TF-IDF + SVD is built for you).

python knn_feats.py train=train.csv test=test.csv target=price id=sample_id folds=folds.csv \
    text=catalog_content k=5,10,20 tf=log out=knn
python knn_feats.py ... emb_tr=img_tr.npy emb_te=img_te.npy out=knn_img
Writes <out>_train.csv and <out>_test.csv; pass them to gbdt_cv.py feats=knn_train.csv:knn_test.csv
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from kv import args, cols, read

a = args({'train': 'train.csv', 'test': 'test.csv', 'target': 'target', 'id': 'sample_id',
          'folds': 'folds.csv', 'text': '', 'emb_tr': '', 'emb_te': '', 'k': '5,10,20', 'tf': 'log',
          'svd': 128, 'out': 'knn', 'seed': 42})
tr, te = read(a['train']), read(a['test'])
fo = read(a['folds'])
fold = tr[a['id']].map(dict(zip(fo[fo.columns[0]], fo['fold']))).values.astype(int)
y = tr[a['target']].values.astype(float)
y = {'log': np.log, 'log1p': np.log1p, 'none': lambda v: v}[a['tf']](y)
ks = [int(k) for k in cols(a['k'])]
K = max(ks)

if a['emb_tr']:
    Etr, Ete = np.load(a['emb_tr']).astype('float32'), np.load(a['emb_te']).astype('float32')
else:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    txt = pd.concat([tr[a['text']], te[a['text']]]).fillna('').astype(str).str.lower()
    Z = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=300000, sublinear_tf=True).fit_transform(txt)
    Z = TruncatedSVD(a['svd'], random_state=a['seed']).fit_transform(Z).astype('float32')
    Etr, Ete = Z[:len(tr)], Z[len(tr):]
Etr, Ete = normalize(Etr), normalize(Ete)


def search(base, q):
    try:
        import faiss
        ix = faiss.IndexFlatIP(base.shape[1]); ix.add(base)
        s, i = ix.search(q, K)
        return 1 - s, i
    except ImportError:
        nn = NearestNeighbors(n_neighbors=K, metric='cosine', algorithm='brute', n_jobs=-1).fit(base)
        return nn.kneighbors(q)


def stats(d, i, yb):
    yy = yb[i]
    w = 1 / (d + 1e-4)
    r = {'y1': yy[:, 0], 'd1': d[:, 0]}
    for k in ks:
        r[f'mean{k}'] = yy[:, :k].mean(1)
        r[f'med{k}'] = np.median(yy[:, :k], 1)
        r[f'std{k}'] = yy[:, :k].std(1)
        r[f'wmean{k}'] = (yy[:, :k] * w[:, :k]).sum(1) / w[:, :k].sum(1)
        r[f'dmean{k}'] = d[:, :k].mean(1)
    return pd.DataFrame(r)


parts = []
for f in range(fold.max() + 1):
    t, v = np.where(fold != f)[0], np.where(fold == f)[0]
    d, i = search(Etr[t], Etr[v])
    s = stats(d, i, y[t]); s.index = v
    parts.append(s)
    print('fold', f, 'done')
ftr = pd.concat(parts).sort_index()
d, i = search(Etr, Ete)
fte = stats(d, i, y)
pre = a['out'] + '_'
ftr.columns = [pre + c for c in ftr.columns]; fte.columns = ftr.columns
ftr.insert(0, a['id'], tr[a['id']].values); fte.insert(0, a['id'], te[a['id']].values)
ftr.to_csv(a['out'] + '_train.csv', index=False); fte.to_csv(a['out'] + '_test.csv', index=False)
c = np.corrcoef(ftr[pre + f'med{ks[0]}'], y)[0, 1]
print(f'corr({pre}med{ks[0]}, target) OOF = {c:.3f}  -> wrote {a["out"]}_train.csv / _test.csv')
