"""Turn semi-structured catalog text into numeric features (quantity, unit, pack, brand, flags).

python parse_catalog.py inp=train.csv col=catalog_content id=sample_id out=feat_train.csv
Run on train and test separately with the same settings. Merge on id.

Handles the 2025 layout (Item Name / Bullet Point N / Product Description / Value / Unit / IPQ)
and free text ("12 oz", "Pack of 6", "2 x 500 ml"). Extend UNIT and KW for a new domain.
"""
import re
import numpy as np
import pandas as pd
from kv import args, read

# unit -> (base, factor). base: g (mass), ml (volume), ct (count), cm (length), w, v
UNIT = {
    'mg': ('g', .001), 'g': ('g', 1), 'gram': ('g', 1), 'grams': ('g', 1), 'gm': ('g', 1), 'gms': ('g', 1),
    'kg': ('g', 1000), 'kilogram': ('g', 1000), 'oz': ('g', 28.3495), 'ounce': ('g', 28.3495),
    'ounces': ('g', 28.3495), 'lb': ('g', 453.592), 'lbs': ('g', 453.592), 'pound': ('g', 453.592),
    'pounds': ('g', 453.592),
    'ml': ('ml', 1), 'millilitre': ('ml', 1), 'milliliter': ('ml', 1), 'l': ('ml', 1000), 'liter': ('ml', 1000),
    'litre': ('ml', 1000), 'fl oz': ('ml', 29.5735), 'fl. oz': ('ml', 29.5735), 'floz': ('ml', 29.5735),
    'fz': ('ml', 29.5735), 'fluid ounce': ('ml', 29.5735), 'fluid ounces': ('ml', 29.5735),
    'gallon': ('ml', 3785.41), 'gal': ('ml', 3785.41), 'quart': ('ml', 946.353), 'qt': ('ml', 946.353),
    'pint': ('ml', 473.176), 'cup': ('ml', 236.6),
    'count': ('ct', 1), 'ct': ('ct', 1), 'pcs': ('ct', 1), 'pieces': ('ct', 1), 'piece': ('ct', 1),
    'each': ('ct', 1), 'units': ('ct', 1), 'tablets': ('ct', 1), 'capsules': ('ct', 1), 'bags': ('ct', 1),
    'cm': ('cm', 1), 'mm': ('cm', .1), 'm': ('cm', 100), 'inch': ('cm', 2.54), 'inches': ('cm', 2.54),
    'in': ('cm', 2.54), 'ft': ('cm', 30.48), 'foot': ('cm', 30.48), 'feet': ('cm', 30.48),
    'w': ('w', 1), 'watt': ('w', 1), 'watts': ('w', 1), 'kw': ('w', 1000), 'v': ('v', 1), 'volt': ('v', 1),
    'volts': ('v', 1),
}
KW = ['organic', 'premium', 'gourmet', 'gift', 'bulk', 'refill', 'value pack', 'family size', 'natural',
      'gluten free', 'vegan', 'kosher', 'sugar free', 'non-gmo', 'imported', 'handmade', 'variety pack',
      'travel size', 'sample', 'case', 'wholesale', 'luxury', 'professional']

u_alt = '|'.join(sorted((re.escape(k) for k in UNIT), key=len, reverse=True))
RX_Q = re.compile(r'(\d+(?:[.,]\d+)?)\s*(' + u_alt + r')\b\.?', re.I)
RX_PK = re.compile(r'(?:pack|case|set|box|bundle)\s*of\s*(\d+)|(\d+)\s*[- ]?(?:pack|pk)\b|(\d+)\s*[x×]\s*\d', re.I)
RX_VAL = re.compile(r'^\s*Value:\s*([\d.,]+)', re.I | re.M)
RX_UNIT = re.compile(r'^\s*Unit:\s*(.+)$', re.I | re.M)
RX_IPQ = re.compile(r'(?:item pack quantity|ipq)\s*[:=]?\s*(\d+)', re.I)
RX_NAME = re.compile(r'Item Name:\s*(.+)', re.I)
RX_BUL = re.compile(r'Bullet Point', re.I)


def num(s):
    try:
        return float(str(s).replace(',', ''))
    except ValueError:
        return np.nan


def base(v, u):
    u = str(u).strip().lower().rstrip('.')
    if u in UNIT:
        b, f = UNIT[u]
        return b, v * f
    return 'other', v


def feat(t):
    t = '' if not isinstance(t, str) else t
    lo = t.lower()
    r = {}
    nm = RX_NAME.search(t)
    name = nm.group(1).split('\n')[0] if nm else t.split('\n')[0]
    r['brand'] = ' '.join(name.split()[:2]).lower() if name else ''
    r['brand1'] = name.split()[0].lower() if name.split() else ''
    v = RX_VAL.search(t); u = RX_UNIT.search(t)
    r['val'] = num(v.group(1)) if v else np.nan
    r['unit'] = u.group(1).strip().lower() if u else ''
    qs = [(num(x), y.lower()) for x, y in RX_Q.findall(name)] or [(num(x), y.lower()) for x, y in RX_Q.findall(t)]
    r['n_qty_mentions'] = len(qs)
    nb, nq = base(*qs[0]) if qs else ('other', np.nan)          # quantity written in the title
    vb, vq = base(r['val'], r['unit']) if (not np.isnan(r['val']) and r['unit']) else ('other', np.nan)
    pk = [int(next(g for g in m if g)) for m in RX_PK.findall(name + ' ' + t[:400])]
    ipq = RX_IPQ.search(t)
    r['ipq'] = int(ipq.group(1)) if ipq else np.nan
    r['pack'] = float(max(pk)) if pk else (r['ipq'] if not np.isnan(r['ipq']) else 1.0)
    r['name_qty'] = nq
    r['name_total'] = nq * r['pack'] if not np.isnan(nq) else np.nan
    r['val_base'] = vq                                            # structured Value, unit-normalised
    # structured Value wins when present (check on EDA whether it is per-unit or total!)
    r['qty_base'] = vb if not np.isnan(vq) else nb
    r['total_qty'] = vq if not np.isnan(vq) else r['name_total']
    r['log_total_qty'] = np.log1p(r['total_qty']) if not np.isnan(r['total_qty']) else np.nan
    r['val_over_name'] = vq / r['name_total'] if (not np.isnan(vq) and r['name_total']) else np.nan
    r['n_bullets'] = len(RX_BUL.findall(t))
    r['len_chars'] = len(t)
    r['len_words'] = len(t.split())
    r['n_digits'] = sum(ch.isdigit() for ch in t)
    r['name_len'] = len(name)
    for k in KW:
        r['kw_' + k.replace(' ', '_').replace('-', '_')] = int(k in lo)
    return r


if __name__ == '__main__':
    a = args({'inp': 'train.csv', 'col': 'catalog_content', 'id': 'sample_id', 'out': 'feat.csv'})
    df = read(a['inp'])
    F = pd.DataFrame([feat(x) for x in df[a['col']]])
    F.insert(0, a['id'], df[a['id']].values)
    F.to_csv(a['out'], index=False)
    cov = F[['total_qty', 'name_qty', 'val', 'pack']].notna().mean().round(3).to_dict()
    print('rows', len(F), '| coverage', cov, '| qty_base mix', F['qty_base'].value_counts().head(6).to_dict())
    print('wrote', a['out'], '| low qty coverage means: extend UNIT/regex, or use an LLM extractor (references/techniques.md C2)')
