"""Re-run norm.norm_addr() (with the new city->region fallback) on RAW French test S2/S3 addresses and diff against
the currently-normalized `state` field. Must be purely additive: every row that already had a state keeps the exact
same value; only previously-empty rows may change, and only to a non-empty value.
python ana/verify_city_fallback.py     (ER_WORK = work dir)"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import polars as pl  # noqa: E402
import norm  # noqa: E402

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')
print('FR_CITY_REGION loaded:', len(norm.FR_CITY_REGION), 'entries', flush=True)

for k in (2, 3):
    raw = pl.read_parquet(f'{W}/raw_test_s{k}.parquet', columns=['id', 'addr', 'country']).filter(pl.col('country') == 'France')
    cur = pl.read_parquet(f'{W}/n_test_s{k}.parquet', columns=['id', 'state']).rename({'state': 'old_state'})
    new_states = [norm.norm_addr(a, 'France')['state'] for a in raw['addr'].to_list()]
    d = raw.with_columns(new_state=pl.Series(new_states)).join(cur, on='id')
    changed_when_had_one = d.filter((pl.col('old_state').fill_null('') != '') & (pl.col('new_state') != pl.col('old_state').fill_null('')))
    newly_filled = d.filter((pl.col('old_state').fill_null('') == '') & (pl.col('new_state') != ''))
    still_empty = d.filter((pl.col('old_state').fill_null('') == '') & (pl.col('new_state') == ''))
    print(f'S{k}: {d.height} France rows | changed an EXISTING state: {changed_when_had_one.height} (must be 0) | '
          f'newly filled: {newly_filled.height} | still empty: {still_empty.height}', flush=True)
    if changed_when_had_one.height:
        print('  BUG sample:', changed_when_had_one.select('addr', 'old_state', 'new_state').head(5))
    print('  sample newly filled:')
    for a, ns in zip(newly_filled['addr'].to_list()[:5], newly_filled['new_state'].to_list()[:5]):
        print(f'    {a!r} -> {ns!r}')

# non-France sanity: nothing changes for US/India at all (pass the same tdict.json the real pipeline uses)
import json
tdict = json.load(open(f'{W}/tdict.json')) if os.path.exists(f'{W}/tdict.json') else {}
print('tdict entries', len(tdict), flush=True)
for k in (2, 3):
    raw = pl.read_parquet(f'{W}/raw_test_s{k}.parquet', columns=['id', 'addr', 'country']).filter(pl.col('country') != 'France').sample(20000, seed=0)
    cur = pl.read_parquet(f'{W}/n_test_s{k}.parquet', columns=['id', 'state']).rename({'state': 'old_state'})
    new_states = [norm.norm_addr(a, c, tdict)['state'] for a, c in raw.select('addr', 'country').iter_rows()]
    d = raw.with_columns(new_state=pl.Series(new_states)).join(cur, on='id')
    diff = d.filter(pl.col('new_state') != pl.col('old_state').fill_null(''))
    print(f'US/India S{k} sample (with tdict): {diff.height} of {d.height} changed (must be 0)', flush=True)
    if diff.height:
        print('  sample diffs:', diff.select('addr', 'country', 'old_state', 'new_state').head(5))
