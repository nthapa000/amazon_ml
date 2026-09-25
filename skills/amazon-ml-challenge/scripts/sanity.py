"""Check a submission before upload. Catches the mistakes that waste a submission slot.

python sanity.py sub=test_out.csv test=test.csv id=sample_id col=price pos=1
python sanity.py sub=test_out.csv test=test.csv id=index col=prediction fmt=value_unit units=gram,kilogram,centimetre
  pos=1        every prediction must be > 0 (prices, lengths)
  fmt=value_unit  2024 style: "<float> <unit>" or empty string; units= allowed list
  sample=      organiser's sample_submission.csv to compare column names and order
"""
import re
import sys
import numpy as np
import pandas as pd
from kv import args, cols, read

a = args({'sub': 'test_out.csv', 'test': 'test.csv', 'id': 'sample_id', 'col': '', 'pos': False,
          'fmt': '', 'units': '', 'sample': ''})
s, te = read(a['sub']), read(a['test'])
err, warn = [], []
if a['sample']:
    sm = read(a['sample'])
    if list(sm.columns) != list(s.columns):
        err.append(f'columns {list(s.columns)} != sample {list(sm.columns)}')
col = a['col'] or [c for c in s.columns if c != a['id']][0]
if len(s) != len(te):
    err.append(f'row count {len(s)} != test {len(te)}')
if a['id'] not in s:
    err.append(f'missing id column {a["id"]!r}')
else:
    if s[a['id']].duplicated().any():
        err.append(f'{int(s[a["id"]].duplicated().sum())} duplicate ids')
    miss = set(te[a['id']]) - set(s[a['id']])
    extra = set(s[a['id']]) - set(te[a['id']])
    if miss:
        err.append(f'{len(miss)} test ids missing, e.g. {list(miss)[:3]}')
    if extra:
        err.append(f'{len(extra)} unknown ids, e.g. {list(extra)[:3]}')
v = s[col]
if a['fmt'] == 'value_unit':
    U = set(cols(a['units']))
    rx = re.compile(r'^\d+(\.\d+)?(e[+-]?\d+)?\s+\S.*$', re.I)
    vv = v.fillna('').astype(str)
    bad = vv[(vv != '') & ~vv.str.match(rx)]
    if len(bad):
        err.append(f'{len(bad)} rows not "<float> <unit>", e.g. {bad.head(3).tolist()}')
    if U:
        uu = vv[vv != ''].str.split(n=1).str[1]
        badu = uu[~uu.isin(U)]
        if len(badu):
            err.append(f'{len(badu)} rows with units outside allowed list, e.g. {badu.head(3).tolist()}')
    print(f'empty predictions: {(vv == "").mean():.2%}')
else:
    if v.isna().any():
        err.append(f'{int(v.isna().sum())} NaN predictions')
    num = pd.to_numeric(v, errors='coerce')
    if num.notna().all():
        if not np.isfinite(num).all():
            err.append('inf values present')
        if a['pos'] and (num <= 0).any():
            err.append(f'{int((num <= 0).sum())} non-positive predictions')
        print(num.describe().round(4).to_string())
        if num.nunique() < 5:
            warn.append('fewer than 5 distinct predictions: collapsed model?')
for w in warn:
    print('WARN:', w)
if err:
    print('\n'.join('FAIL: ' + e for e in err))
    sys.exit(1)
print('PASS: submission looks valid')
