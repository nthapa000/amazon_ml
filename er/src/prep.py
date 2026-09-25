"""prep.py raw   : 7 TSVs -> parquet (W/raw_{split}_s{k}.parquet, W/gt.parquet)
   prep.py norm  : apply norm.py (+ learned dict) in parallel -> W/n_{split}_s{k}.parquet
Env: ER_DATA (dataset dir), ER_WORK (work dir)."""
import os
import sys
import json
import time
from multiprocessing import Pool

import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
import norm  # noqa: E402

D = os.environ.get('ER_DATA', '/csehome/m25csa019/amazon_ml/student_resource/dataset')
W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')
SPLITS = ['train', 'test']


def read_tsv(p):
    return pl.read_csv(p, separator='\t', quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)


def do_raw():
    os.makedirs(W, exist_ok=True)
    for sp in SPLITS:
        for k in (1, 2, 3):
            df = read_tsv(f'{D}/{sp}/{sp}_source{k}.tsv').fill_null('')
            df = df.rename({'entity_id': 'id', 'business_name': 'name', 'business_address': 'addr'})
            df.write_parquet(f'{W}/raw_{sp}_s{k}.parquet')
            print(sp, k, df.shape, flush=True)
    gt = read_tsv(f'{D}/train/train_ground_truth.tsv').fill_null('')
    gt = gt.rename({'source1_entity_id': 's1', 'matched_entity_ids': 'm'})
    gt.write_parquet(f'{W}/gt_raw.parquet')
    pairs = (gt.filter(pl.col('m') != '').with_columns(pl.col('m').str.split(',')).explode('m')
             .rename({'m': 'id2'}))
    pairs.write_parquet(f'{W}/gt_pairs.parquet')
    print('gt', gt.shape, 'pairs', pairs.shape)


_TD = None


def _init(tdp):
    global _TD
    _TD = json.load(open(tdp)) if tdp and os.path.exists(tdp) else {}


def _work(rows):
    out = []
    for i, n, a, c in rows:
        r = norm.norm_name(n, _TD)
        r.update(norm.norm_addr(a, c, _TD))
        r['id'] = i
        out.append(r)
    return out


def do_norm(nproc=32):
    tdp = f'{W}/tdict.json'
    with Pool(nproc, initializer=_init, initargs=(tdp,)) as pool:
        for sp in SPLITS:
            for k in (1, 2, 3):
                t0 = time.time()
                df = pl.read_parquet(f'{W}/raw_{sp}_s{k}.parquet')
                rows = list(zip(df['id'], df['name'], df['addr'], df['country']))
                ch = [rows[i:i + 20000] for i in range(0, len(rows), 20000)]
                res = [x for part in pool.imap(_work, ch) for x in part]
                nd = pl.DataFrame(res)
                out = df.join(nd, on='id', how='left')
                out.write_parquet(f'{W}/n_{sp}_s{k}.parquet')
                print(sp, k, out.shape, f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'raw':
        do_raw()
    elif cmd == 'norm':
        do_norm(int(os.environ.get('SLURM_CPUS_PER_TASK', 32)))
