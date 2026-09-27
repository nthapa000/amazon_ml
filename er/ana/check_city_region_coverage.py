import os
import sys
import json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import polars as pl  # noqa: E402
from dicts import _city_tok  # noqa: E402

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')
lut = json.load(open(f'{W}/fr_city_region.json'))['France']
print('lookup size', len(lut))

s23 = pl.concat([pl.read_parquet(f'{W}/n_test_s{k}.parquet', columns=['country', 'ad', 'state']) for k in (2, 3)]).filter(pl.col('country') == 'France')
miss = s23.filter(pl.col('state').fill_null('') == '')
print('state-missing S2/S3 rows', miss.height, 'of', s23.height)
covered = 0
for ad in miss['ad'].fill_null('').to_list():
    comps = [p.strip() for p in ad.split(' , ') if p.strip()]
    if _city_tok(comps) in lut:
        covered += 1
print(f'recovered by this lookup: {covered} / {miss.height} ({covered / miss.height:.4f})')
