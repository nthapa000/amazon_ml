"""Create ONE fold file the whole team shares. Never change it mid-competition.

python make_folds.py train=train.csv target=price id=sample_id k=5 task=reg group=
  task=reg  -> stratify on target quantile bins (log-safe)
  task=clf  -> stratify on class
  group=col -> rows sharing this value (e.g. normalised title) never straddle folds
Writes folds.csv with columns [id, fold].
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from kv import args, read

a = args({'train': 'train.csv', 'target': 'target', 'id': '', 'k': 5, 'task': 'reg',
          'group': '', 'seed': 42, 'out': 'folds.csv', 'bins': 10})
tr = read(a['train'])
y = tr[a['target']]
if a['task'] == 'reg':
    z = np.log1p(y.clip(lower=0)) if (y >= 0).all() else y
    s = pd.qcut(z.rank(method='first'), a['bins'], labels=False)
else:
    s = y.astype(str)
fold = np.zeros(len(tr), int)
if a['group']:
    g = tr[a['group']].astype(str).str.lower().str.replace(r'\W+', ' ', regex=True).str.strip()
    sp = StratifiedGroupKFold(a['k'], shuffle=True, random_state=a['seed']).split(tr, s, g)
else:
    sp = StratifiedKFold(a['k'], shuffle=True, random_state=a['seed']).split(tr, s)
for f, (_, v) in enumerate(sp):
    fold[v] = f
ids = tr[a['id']] if a['id'] else pd.Series(np.arange(len(tr)), name='row')
out = pd.DataFrame({ids.name: ids.values, 'fold': fold})
out.to_csv(a['out'], index=False)
print(out['fold'].value_counts().sort_index().to_string())
print('wrote', a['out'])
