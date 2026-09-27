"""Why is `state` empty on ~35% of French S2/S3 records (test s2/s3), and is it recoverable from the provided data alone?
Unsupervised: builds a city->region lookup ONLY from French S1's own (city, region) co-occurrence (S1 has state 100% of
the time), then checks how much of the S2/S3 state-empty gap a city match alone would close -- no test labels, no
external data, same footing as the existing FR_DEP table (rule/co-occurrence from provided records).
python ana/fr_state_gap.py     (ER_WORK = work dir)"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import polars as pl  # noqa: E402

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')

s1 = pl.read_parquet(f'{W}/n_test_s1.parquet', columns=['country', 'ad', 'state']).filter(pl.col('country') == 'France')
s23 = pl.concat([pl.read_parquet(f'{W}/n_test_s{k}.parquet', columns=['country', 'ad', 'state']) for k in (2, 3)]).filter(pl.col('country') == 'France')
print('S1 France', s1.height, 'S2/S3 France', s23.height, flush=True)

miss = s23.filter(pl.col('state').fill_null('') == '')
print(f'S2/S3 missing state: {miss.height} ({miss.height / s23.height:.4f})', flush=True)

# what's actually in the address of the missing-state rows?
comps = miss.with_columns(pl.col('ad').fill_null('').str.split(' , ').alias('c'))
print('rows with a non-empty address at all:', comps.filter(pl.col('ad').fill_null('') != '').height, flush=True)
print('sample raw `ad` of state-missing rows:', flush=True)
for a in miss['ad'].fill_null('(empty)').to_list()[:12]:
    print('  ', repr(a))

# city->region lookup, learned ONLY from S1's own (city, region) co-occurrence (no test labels; S1 always has region)
BAD = {'etage', 'batiment', 'bat', 'appartement', 'appt', 'app', 'rez de chaussee', 'rdc', 'escalier', 'esc',
       'maison des associations', 'immeuble'}


def city_tok(ad):
    """last comma-separated component that isn't a building/floor descriptor and has no digits; else the one before it."""
    parts = [p.strip() for p in ad.split(' , ') if p.strip()]
    for x in reversed(parts):
        if x and not any(ch.isdigit() for ch in x) and x not in BAD and not any(b in x for b in BAD):
            return x
    return ''

s1c = s1.filter(pl.col('state').fill_null('') != '').with_columns(city=pl.col('ad').fill_null('').map_elements(city_tok, return_dtype=pl.String))
purity = s1c.group_by('city').agg(pl.col('state').mode().first().alias('mode'), pl.len().alias('n'),
                                   pl.col('state').n_unique().alias('nun'))
lut = purity.filter((pl.col('n') >= 3) & (pl.col('nun') == 1)).select('city', pl.col('mode').alias('region'))
print(f'\ncity->region lookup entries (n>=3, pure): {lut.height}', flush=True)

miss_city = miss.with_columns(city=pl.col('ad').fill_null('').map_elements(city_tok, return_dtype=pl.String))
hit = miss_city.join(lut, on='city', how='left')
covered = hit.filter(pl.col('region').is_not_null()).height
print(f'state-missing S2/S3 rows whose city matches the S1-learned lookup: {covered} / {miss.height} ({covered / miss.height:.4f})', flush=True)
print('top uncovered city tokens:', Counter(hit.filter(pl.col('region').is_null())['city'].to_list()).most_common(15), flush=True)
