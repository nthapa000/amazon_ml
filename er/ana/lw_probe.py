"""Does the class of a last-word change (noise-inserted word vs ordinary word) predict a match, and does stage 1 already know?
Noise words = last words over-represented on S2/S3 vs S1 within the country (excess >= 0.004, computed from the split's own records,
no labels; ana/noise_words.py). Candidate pairs (stage-1 p >= 0.001) whose names differ only in the LAST word:
  lastswap: same words but the last one replaced | lastadd: S2/S3 = S1 + one word | lastdrop: S2/S3 = S1 minus its last word
bucketed by the class of the changed word. tsim = labelled US/India; test = counts and stage-1 p only (France has no labels).
python ana/lw_probe.py      (ER_WORK = work dir)"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import polars as pl  # noqa: E402
from block import enc  # noqa: E402

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')
TH = 0.004


def recs(sp, ks):
    return pl.concat([pl.read_parquet(f'{W}/n_{sp}_s{k}.parquet', columns=['id', 'country', 'core']) for k in ks]).select(
        enc('id').alias('id'), 'country', pl.col('core').fill_null('').str.split(' ').alias('t'))


def run(sp, pairs_path):
    s1, s23 = recs(sp, (1,)), recs(sp, (2, 3))
    lw = lambda d: d.with_columns(w=pl.col('t').list.last()).filter(pl.col('w').is_not_null() & (pl.col('w') != ''))
    f1 = lw(s1).group_by('country', 'w').len('n1').with_columns(f1=pl.col('n1') / pl.col('n1').sum().over('country'))
    f2 = lw(s23).group_by('country', 'w').len('n2').with_columns(f2=pl.col('n2') / pl.col('n2').sum().over('country'))
    ex = f2.join(f1, on=['country', 'w'], how='left').with_columns(ex=pl.col('f2') - pl.col('f1').fill_null(0)).select('country', 'w', 'ex')
    noisy = ex.filter(pl.col('ex') >= TH)
    for c in sorted(noisy['country'].unique().to_list()):
        print(sp, c, 'noise words', noisy.filter(pl.col('country') == c).sort('ex', descending=True)['w'].to_list(), flush=True)
    p = pl.read_parquet(pairs_path).filter(pl.col('p') >= 0.001)
    d = (p.join(s1.rename({'id': 'id1', 't': 't1'}), on='id1').join(s23.select(pl.col('id').alias('id2'), pl.col('t').alias('t2')), on='id2')
         .filter((pl.col('t1').list.len() >= 2) & (pl.col('t2').list.len() >= 1)))
    n1, n2 = pl.col('t1').list.len(), pl.col('t2').list.len()
    d = d.with_columns(
        kind=pl.when((n1 == n2) & (pl.col('t1').list.head(n1 - 1) == pl.col('t2').list.head(n2 - 1)) & (pl.col('t1').list.last() != pl.col('t2').list.last())).then(pl.lit('lastswap'))
        .when((n2 == n1 + 1) & (pl.col('t2').list.head(n1) == pl.col('t1'))).then(pl.lit('lastadd'))
        .when((n2 == n1 - 1) & (pl.col('t1').list.head(n2) == pl.col('t2'))).then(pl.lit('lastdrop'))
        .otherwise(pl.lit('none')),
        cw=pl.when(n2 >= n1).then(pl.col('t2').list.last()).otherwise(pl.col('t1').list.last()))
    d = d.filter(pl.col('kind') != 'none')
    ls = d.filter(pl.col('kind') == 'lastswap')          # split last-word swaps: garbled word (typo-like) vs a genuinely different word
    from rapidfuzz import fuzz, process
    rt = process.cpdist(ls['t1'].list.last().to_list(), ls['t2'].list.last().to_list(), scorer=fuzz.ratio, workers=8)
    ls = ls.with_columns(garble=pl.Series(rt) >= 50)
    d = pl.concat([ls, d.filter(pl.col('kind') != 'lastswap').with_columns(garble=pl.lit(False))]).join(noisy.select('country', pl.col('w').alias('cw'), pl.lit(True).alias('nz')), on=['country', 'cw'], how='left').with_columns(
        cls=pl.when(pl.col('garble')).then(pl.lit('garbled')).when(pl.col('nz').fill_null(False)).then(pl.lit('noise-word')).otherwise(pl.lit('ordinary')))
    band = (pl.col('p') > 0.1) & (pl.col('p') < 0.9)
    aggs = [pl.len().alias('pairs'), pl.col('p').mean().alias('mean_p'), band.sum().alias('n_band'), pl.col('p').filter(band).mean().alias('p_in_band')]
    if 'y' in d.columns:
        aggs += [pl.col('y').mean().alias('match_rate'), pl.col('y').filter(band).mean().alias('match_in_band')]
    with pl.Config(tbl_rows=40, tbl_cols=12, tbl_width_chars=200):
        print(f'== {sp}: candidate pairs (p >= 0.001) whose names differ only by a swapped last word')
        print(d.filter(pl.col('kind') == 'lastswap').group_by('country', 'cls').agg(aggs).sort('country', 'cls'), flush=True)


run('tsim', f'{W}/oof_b1ka.parquet')
run('test', f'{W}/pte_b1ka.parquet')
