"""Graph-based candidate expansion (recall).

A missed S2/S3 record often has a sibling record of the same business that we already matched confidently.
We enrich every S1 with the fields of its confident matches ("pseudo-S1" records) and re-run blocking of the
still-unassigned S2/S3 records against S1 + pseudo-S1. New pairs (S1, x) are returned, capped by key score.

python expand.py split=tsim pred=oof_u3 thr=0.9 base=cand_tsim_k100 k1=20 k2=5
-> W/exp_{split}.parquet (id1, id2, ks, kbits, nk, r1, r2, e_sib)   (+ recall report when labels exist)
e_sib = 1 if the pair came via a sibling (pseudo) record.
"""
import os
import sys
import time

import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
from block import load, gen_pairs, cap, enc, gt_path  # noqa: E402
from decide import one2one  # noqa: E402
from metric import macro_fbeta_arr  # noqa: E402

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')


def main(kv):
    split, thr = kv['split'], float(kv.get('thr', 0.9))
    k1, k2 = int(kv.get('k1', 20)), int(kv.get('k2', 5))
    pred = pl.read_parquet(f'{W}/{kv["pred"]}.parquet', columns=['id1', 'id2', 'p'])
    conf = one2one(pred, 0.2).filter(pl.col('p') >= thr).select('id1', 'id2')
    base = pl.read_parquet(f'{W}/{kv["base"]}.parquet', columns=['id1', 'id2'])
    print('confident pairs', conf.height, flush=True)
    ctrs = pl.read_parquet(f'{W}/n_{split}_s1.parquet', columns=['country'])['country'].unique().sort().to_list()
    out = []
    for c in ctrs:
        t0 = time.time()
        rec = load(split, c)
        s1 = rec.filter(pl.col('side') == 0)
        s23 = rec.filter(pl.col('side') == 1)
        assigned = conf.select(pl.col('id2').alias('id'))
        # pseudo-S1: confident S2/S3 records re-labelled as side 0 with temporary negative ids
        ps = (s23.join(conf.rename({'id2': 'id'}), on='id').with_row_index('r')
              .with_columns(pid=-(pl.col('r').cast(pl.Int64) + 1), side=pl.lit(0, pl.Int8)))
        pmap = ps.select('pid', 'id1')
        ps = ps.drop('id', 'r', 'id1').rename({'pid': 'id'}).select(rec.columns)
        un = s23.join(assigned, on='id', how='anti')
        pr = gen_pairs(pl.concat([s1, ps, un]), c)
        # map pseudo ids back to their S1; keep only pairs with unassigned records
        pr = (pr.join(pmap.rename({'pid': 'id1', 'id1': 'own'}), on='id1', how='left')
              .with_columns(e_sib=pl.col('own').is_not_null().cast(pl.Int8),
                            id1=pl.coalesce('own', 'id1')).drop('own'))
        pr = pr.group_by('id1', 'id2').agg(pl.col('ks').sum(), pl.col('kbits').max(), pl.col('nk').sum().cast(pl.Int16),
                                           pl.col('e_sib').max())
        pr = pr.join(base, on=['id1', 'id2'], how='anti')
        pr = cap(pr, k1, k2)
        out.append(pr)
        print(c, 'new pairs', pr.height, f'{time.time() - t0:.0f}s', flush=True)
    new = pl.concat(out)
    new.write_parquet(f'{W}/{kv.get("out", "exp_" + split)}.parquet')
    if split != 'test':
        gt = pl.read_parquet(gt_path(split)).select(enc('s1').alias('id1'), enc('id2').alias('id2'))
        miss = gt.join(base, on=['id1', 'id2'], how='anti')
        rec_new = miss.join(new, on=['id1', 'id2'], how='semi').height
        print(f'GT pairs {gt.height}  missed by base {miss.height}  recovered {rec_new} '
              f'({rec_new / max(miss.height, 1):.3f})  new pairs {new.height}  '
              f'new pos rate {rec_new / max(new.height, 1):.4f}', flush=True)
        s1 = pl.read_parquet(f'{W}/n_{split}_s1.parquet', columns=['id']).select(enc('id').alias('s1'))
        both = pl.concat([base, new.select('id1', 'id2')])
        hit = gt.join(both, on=['id1', 'id2'], how='semi')
        per = (s1.join(gt.group_by('id1').len('nt').rename({'id1': 's1'}), on='s1', how='left')
               .join(hit.group_by('id1').len('nh').rename({'id1': 's1'}), on='s1', how='left').fill_null(0))
        f, _ = macro_fbeta_arr(per['nt'], per['nh'], per['nh'])
        print(f'new ceiling F0.5 {f:.4f}  recall {hit.height / gt.height:.4f}', flush=True)


if __name__ == '__main__':
    main(dict(a.split('=', 1) for a in sys.argv[1:]))
