"""Integrate ANN (embedding) candidates into the kx candidate set -> "ka" = kx + ANN top-K new pairs.

python annka.py cand  split=tsim k=3     -> cand_ann_{split} (base schema: key evidence absent: ks=0, kbits=0, nk=0,
                                            r1=r2=-1) and annf_ann_{split} (id1, id2, is_ann=1, ann_s, ann_r)
python annka.py union split=tsim         -> {xf,f2,sib,annf}_{split}_ka extras (kx rows + ann rows),
                                            cand_{split}_ka (id1,id2) and f_cand_{split}_ka/ (links to kx + ann parts)
python annka.py s1 tsim=w1 ann=w1ann out=w1ka  -> oof_/pte_{out} = stage-1 scores of kx pairs (w1) + ann pairs (w1ann)
"""
import glob
import os
import sys

import polars as pl

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')
cmd = sys.argv[1]
kv = dict(a.split('=', 1) for a in sys.argv[2:])


def cand(sp, k):
    a = pl.read_parquet(f'{W}/ann_{sp}.parquet').filter(pl.col('ann_r') <= k)
    c = a.select('id1', 'id2', ks=pl.lit(0.0, pl.Float32), kbits=pl.lit(0, pl.Int16), nk=pl.lit(0, pl.Int16),
                 r1=pl.lit(-1, pl.Int32), r2=pl.lit(-1, pl.Int32))
    c.write_parquet(f'{W}/cand_ann_{sp}.parquet')
    a.select('id1', 'id2', is_ann=pl.lit(1, pl.Int8), ann_s=pl.col('ann_s').cast(pl.Float32),
             ann_r=pl.col('ann_r').cast(pl.Int16)).write_parquet(f'{W}/annf_ann_{sp}.parquet')
    print(sp, 'ann candidate pairs', c.height)


def union(sp):
    kx = pl.read_parquet(f'{W}/cand_{sp}_kx.parquet', columns=['id1', 'id2'])
    an = pl.read_parquet(f'{W}/cand_ann_{sp}.parquet', columns=['id1', 'id2'])
    pl.concat([kx, an]).write_parquet(f'{W}/cand_{sp}_ka.parquet')
    for e in ('xf', 'f2'):
        pl.concat([pl.read_parquet(f'{W}/{e}_{sp}_kx.parquet'), pl.read_parquet(f'{W}/{e}_ann_{sp}.parquet')],
                  how='diagonal_relaxed').write_parquet(f'{W}/{e}_{sp}_ka.parquet')
    sib = pl.read_parquet(f'{W}/sib_{sp}.parquet')
    pl.concat([sib, an.with_columns(is_exp=pl.lit(0, pl.Int8), e_sib=pl.lit(0, pl.Int8))],
              how='diagonal_relaxed').write_parquet(f'{W}/sib_{sp}_ka.parquet')
    af = pl.read_parquet(f'{W}/annf_ann_{sp}.parquet')
    pl.concat([kx.with_columns(is_ann=pl.lit(0, pl.Int8)), af], how='diagonal_relaxed') \
        .write_parquet(f'{W}/annf_{sp}_ka.parquet')
    d = f'{W}/f_cand_{sp}_ka'
    os.makedirs(d, exist_ok=True)
    for p in sorted(glob.glob(f'{W}/f_cand_{sp}_kx/*.parquet')):
        os.symlink(os.path.realpath(p), f'{d}/{os.path.basename(p)}')
    for p in sorted(glob.glob(f'{W}/f_cand_ann_{sp}/p_*.parquet')):
        os.symlink(p, f'{d}/n_{os.path.basename(p)}')
    print(sp, 'union pairs', kx.height + an.height, 'parts', len(os.listdir(d)))


def s1(a, b, out):
    for kind in ('oof', 'pte'):
        x = pl.read_parquet(f'{W}/{kind}_{a}.parquet')
        y = pl.read_parquet(f'{W}/{kind}_{b}.parquet')
        pl.concat([x, y.select(x.columns)], how='vertical_relaxed').write_parquet(f'{W}/{kind}_{out}.parquet')
        print(kind, x.height, '+', y.height)


if cmd == 'cand':
    cand(kv['split'], int(kv.get('k', 3)))
elif cmd == 'union':
    union(kv['split'])
else:
    s1(kv['tsim'], kv['ann'], kv['out'])
