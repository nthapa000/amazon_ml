"""Export the Qwen3-Reranker-4B cross-encoder (ceq) scores as feature `q4` for a teammate's stage 2. No retraining.

python ana/q4_export.py prep    -> OUT/work: q4_b1ka_{train,test}_h{0,1}.parquet (ceq half files, column ceq -> q4)
                                   + oof_b1ka.parquet = tsim band rows only (id1, id2, y, p), so the merge's AUC check stays small
ER_WORK=OUT/work python src/ce.py merge=1 tag=b1ka name=q4   -> OUT/work/q4_b1ka_{train,test}.parquet
python ana/q4_export.py check   -> verifies counts / dups / nulls / OOF split / AUC, copies the two files to OUT/upload
python ana/q4_export.py fr      -> OUT/ceqf/upload/q4f_b1ka_test.parquet: the ceqf (French-continued ceq) test file, column q4,
                                   a drop-in replacement for q4_b1ka_test at predict time (ceqf has no OOF: its train file is ceq's)
Reads only from SRC (the final work dir) and never writes there.
"""
import os
import sys

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from block import enc  # noqa: E402

SRC = '/scratch/m25csa019/er_final'
OUT = os.environ.get('Q4_OUT', '/scratch/m25csa019/q4_export')
WK = f'{OUT}/work'
LO, HI = 0.003, 0.997                      # ce.py split=tsim tag=b1ka lo=0.003 hi=0.997
N_TR, N_TE = 2_292_757, 3_067_262


def band_rows(name, cols):
    """stream W/<name>.parquet row group by row group, keep LO < p < HI (the 190M-row file never sits in memory); float32 compare as polars does in ce.py"""
    f, out = pq.ParquetFile(f'{SRC}/{name}.parquet'), []
    for b in f.iter_batches(columns=cols, batch_size=1 << 20):
        out.append(b.filter(pc.and_(pc.greater(b['p'], pa.scalar(LO, pa.float32())), pc.less(b['p'], pa.scalar(HI, pa.float32())))))
    return pl.from_arrow(pa.Table.from_batches(out))


def prep():
    os.makedirs(WK, exist_ok=True)
    for s in ('train', 'test'):
        for h in (0, 1):
            pl.read_parquet(f'{SRC}/ceq_b1ka_{s}_h{h}.parquet').rename({'ceq': 'q4'}).write_parquet(f'{WK}/q4_b1ka_{s}_h{h}.parquet')
    ob = band_rows('oof_b1ka', ['id1', 'id2', 'y', 'p'])
    ob.write_parquet(f'{WK}/oof_b1ka.parquet')
    band_rows('pte_b1ka', ['id1', 'id2', 'p']).select('id1', 'id2').write_parquet(f'{WK}/pte_band_ids.parquet')
    print('prep: oof band rows', ob.height, 'pos', int(ob['y'].sum()), flush=True)


def check():
    tr = pl.read_parquet(f'{WK}/q4_b1ka_train.parquet')
    te = pl.read_parquet(f'{WK}/q4_b1ka_test.parquet')
    ok = True

    def rep(msg, cond):
        nonlocal ok
        ok &= bool(cond)
        print(('PASS ' if cond else 'FAIL ') + msg, flush=True)

    for nm, d, n in (('train', tr, N_TR), ('test', te, N_TE)):
        rep(f'{nm} columns {d.columns} dtypes {[str(t) for t in d.dtypes]}', d.columns == ['id1', 'id2', 'q4']
            and d.schema['id1'] == pl.Int64 and d.schema['id2'] == pl.Int64)
        rep(f'{nm} rows {d.height:,} (expected {n:,})', d.height == n)
        rep(f'{nm} duplicate (id1,id2): {d.height - d.select("id1", "id2").unique().height}', d.select('id1', 'id2').is_unique().all())
        rep(f'{nm} q4 nulls {d["q4"].null_count()} NaN {int(d["q4"].is_nan().sum())} inf {int(d["q4"].is_infinite().sum())}',
            d['q4'].null_count() == 0 and not d['q4'].is_nan().any() and not d['q4'].is_infinite().any())
        rep(f'{nm} id1 all S1 (id//1e10==1), id2 all S2/S3', (d['id1'] // 10_000_000_000 == 1).all()
            and (d['id2'] // 10_000_000_000).is_in([2, 3]).all())
        print(f'     {nm} q4 min {d["q4"].min():.3f} mean {d["q4"].mean():.3f} max {d["q4"].max():.3f}', flush=True)

    # coverage: exactly the tsim / test band of stage-1 p
    ob = pl.read_parquet(f'{WK}/oof_b1ka.parquet')
    tb = pl.read_parquet(f'{WK}/pte_band_ids.parquet')
    rep(f'train pairs == tsim band pairs ({ob.height:,})', ob.height == tr.height
        and tr.join(ob, on=['id1', 'id2'], how='anti').height == 0)
    rep(f'test pairs == test band pairs ({tb.height:,})', tb.height == te.height
        and te.join(tb, on=['id1', 'id2'], how='anti').height == 0)

    # out-of-fold: rows of train_h{h} are S1 with hash%2 == h; the half-h model trained only on S1 with hash%2 != h (ce.py main)
    h = [pl.read_parquet(f'{WK}/q4_b1ka_train_h{k}.parquet') for k in (0, 1)]
    for k in (0, 1):
        hk = h[k].select((pl.col('id1').hash(seed=5) % 2).cast(pl.Int8).alias('h'))['h']
        rep(f'train_h{k}: {h[k].height:,} rows, all S1 in fold {k} (model h{k} trained on fold {1 - k})', (hk == k).all())
    s0, s1 = set(h[0]['id1'].unique().to_list()), set(h[1]['id1'].unique().to_list())
    rep(f'S1 sets of the two OOF halves are disjoint ({len(s0):,} + {len(s1):,} S1)', not (s0 & s1))
    rep('train == concat(train_h0, train_h1)', tr.equals(pl.concat(h)))

    # test = mean of the two half models
    t = [pl.read_parquet(f'{WK}/q4_b1ka_test_h{k}.parquet') for k in (0, 1)]
    m = t[0].join(t[1].rename({'q4': 'b'}), on=['id1', 'id2']).join(te.rename({'q4': 'm'}), on=['id1', 'id2'])
    d = np.abs(((m['q4'].to_numpy().astype(np.float64) + m['b'].to_numpy()) / 2) - m['m'].to_numpy()).max()
    rep(f'test q4 == (h0 + h1) / 2 on all {m.height:,} rows (max abs diff {d:.2e})', m.height == te.height and d < 1e-5)
    rep('test halves score the same pair set', t[0].select('id1', 'id2').sort('id1', 'id2').equals(t[1].select('id1', 'id2').sort('id1', 'id2')))

    # identical to the ceq feature our own stage 2 (fin5) used
    for s, d_ in (('train', tr), ('test', te)):
        c = pl.read_parquet(f'{SRC}/ceq_b1ka_{s}.parquet').join(d_, on=['id1', 'id2'], how='inner')
        rep(f'{s} q4 identical to {SRC}/ceq_b1ka_{s}.parquet', c.height == d_.height and (c['ceq'] == c['q4']).all())

    # band AUC against ground truth (label from oof_b1ka.y and, independently, from gt_pairs_tsim)
    j = tr.join(ob, on=['id1', 'id2'])
    gt = pl.read_parquet(f'{SRC}/gt_pairs_tsim.parquet').select(enc('s1').alias('id1'), enc('id2').alias('id2')).unique().with_columns(g=pl.lit(1, pl.Int8))
    j = j.join(gt, on=['id1', 'id2'], how='left').with_columns(pl.col('g').fill_null(0))
    rep(f'oof y == gt_pairs_tsim membership on all band rows (pos {int(j["y"].sum()):,})', (j['y'] == j['g']).all())
    auc, auc_h = roc_auc_score(j['g'], j['q4']), [roc_auc_score(x.join(j, on=['id1', 'id2'])['g'], x['q4']) for x in h]
    print(f'BAND AUC q4 vs ground truth: {auc:.5f}  (half0 {auc_h[0]:.5f}, half1 {auc_h[1]:.5f}; stage-1 p {roc_auc_score(j["g"], j["p"]):.5f})', flush=True)
    rep('band AUC ~0.985', 0.98 < auc < 0.99)

    if ok:
        os.makedirs(f'{OUT}/upload', exist_ok=True)
        for s in ('train', 'test'):
            os.system(f'cp {WK}/q4_b1ka_{s}.parquet {OUT}/upload/')
        print('ALL CHECKS PASS ->', f'{OUT}/upload', flush=True)
    else:
        print('SOME CHECK FAILED: nothing copied to upload/', flush=True)


def fr():
    """ceqf = ceq halves continued on the French self-training set; merge(unseen=1) patched only unseen-country test pairs"""
    od = f'{OUT}/ceqf/upload'
    q = pl.read_parquet(f'{WK}/q4_b1ka_test.parquet')
    f = pl.read_parquet(f'{SRC}/ceqf_b1ka_test.parquet').rename({'ceqf': 'q4'})
    ok = True

    def rep(msg, cond):
        nonlocal ok
        ok &= bool(cond)
        print(('PASS ' if cond else 'FAIL ') + msg, flush=True)

    rep(f'columns {f.columns} dtypes {[str(t) for t in f.dtypes]}', f.columns == ['id1', 'id2', 'q4']
        and f.schema['id1'] == pl.Int64 and f.schema['id2'] == pl.Int64)
    rep(f'rows {f.height:,} (expected {N_TE:,})', f.height == N_TE)
    rep(f'duplicate (id1,id2): {f.height - f.select("id1", "id2").unique().height}', f.select('id1', 'id2').is_unique().all())
    rep(f'q4 nulls {f["q4"].null_count()} NaN {int(f["q4"].is_nan().sum())} inf {int(f["q4"].is_infinite().sum())}',
        f['q4'].null_count() == 0 and not f['q4'].is_nan().any() and not f['q4'].is_infinite().any())

    ct = pl.read_parquet(f'{SRC}/n_test_s1.parquet', columns=['id', 'country']).select(enc('id').alias('id1'), 'country')
    m = (q.join(f.rename({'q4': 'f'}), on=['id1', 'id2'], how='full', coalesce=True).join(ct, on='id1', how='left')
         .with_columns(fr=pl.col('country') == 'France'))
    rep(f'same pair set as q4_b1ka_test ({m.height:,} rows, none missing on either side)', m.height == N_TE
        and m['q4'].null_count() == 0 and m['f'].null_count() == 0)
    nf = m.filter('fr')
    rep(f'non-France rows identical to q4_b1ka_test ({m.height - nf.height:,})', m.filter(~pl.col('fr')).select((pl.col('q4') == pl.col('f')).all()).item())
    rep(f'France rows: {nf.height:,}, changed {int((nf["q4"] != nf["f"]).sum()):,}', nf.height == 908_542)
    h = [pl.read_parquet(f'{SRC}/ceqf_b1ka_test_h{k}.parquet').rename({'ceqf': f'h{k}'}) for k in (0, 1)]
    hh = nf.join(h[0], on=['id1', 'id2']).join(h[1], on=['id1', 'id2'])
    d = (((hh['h0'].to_numpy().astype(np.float64) + hh['h1'].to_numpy()) / 2) - hh['f'].to_numpy())
    rep(f'France rows == (ceqf h0 + h1) / 2 on {hh.height:,} rows (max abs diff {np.abs(d).max():.2e})', hh.height == nf.height and np.abs(d).max() < 1e-5)
    tr = pl.read_parquet(f'{SRC}/ceqf_b1ka_train.parquet').join(pl.read_parquet(f'{WK}/q4_b1ka_train.parquet'), on=['id1', 'id2'])
    rep(f'ceqf train file == q4_b1ka_train on all {tr.height:,} rows (no separate train file needed)',
        tr.height == N_TR and (tr['ceqf'] == tr['q4']).all())

    a, b = nf['q4'].to_numpy(), nf['f'].to_numpy()
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    print(f'France test pairs, q4 (ceq) vs q4f (ceqf): pearson {np.corrcoef(a, b)[0, 1]:.4f}  spearman '
          f'{nf.select(pl.corr("q4", "f", method="spearman")).item():.4f}', flush=True)
    for nm, x in (('ceq ', a), ('ceqf', b)):
        print(f'     {nm} mean {x.mean():7.3f}  q{qs} {np.round(np.quantile(x, qs), 3).tolist()}  share logit>0 {(x > 0).mean():.4f}', flush=True)
    us = m.filter(~pl.col('fr'))['q4'].to_numpy()
    print(f'     US+India test pairs (unchanged) mean {us.mean():.3f}  share logit>0 {(us > 0).mean():.4f}', flush=True)

    if ok:
        os.makedirs(od, exist_ok=True)
        f.write_parquet(f'{od}/q4f_b1ka_test.parquet')
        print('ALL CHECKS PASS ->', f'{od}/q4f_b1ka_test.parquet', flush=True)
    else:
        print('SOME CHECK FAILED: nothing written', flush=True)


if __name__ == '__main__':
    {'prep': prep, 'check': check, 'fr': fr}[sys.argv[1]]()
