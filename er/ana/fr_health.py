"""Unsupervised country health-check on the provided test records: field-extraction completeness and structural
stats, to find gaps a normalization/feature fix could close. No labels used for completeness stats; any candidate
signal is then checked against labelled US/India before being trusted on France.
python ana/fr_health.py     (ER_WORK = work dir)"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import polars as pl  # noqa: E402
from block import enc  # noqa: E402

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')
C = ['id', 'country', 'name', 'addr', 'core', 'ad', 'hn', 'pc', 'state']

with pl.Config(tbl_rows=30, tbl_cols=15, tbl_width_chars=220):
    for sp in ('train', 'test'):
        for src in (1, 2, 3):
            d = pl.read_parquet(f'{W}/n_{sp}_s{src}.parquet', columns=C)
            r = d.group_by('country').agg(
                n=pl.len(),
                empty_addr=(pl.col('addr').fill_null('') == '').mean(),
                has_hn=(pl.col('hn').fill_null('') != '').mean(),
                has_pc=(pl.col('pc').fill_null('') != '').mean(),
                has_state=(pl.col('state').fill_null('') != '').mean(),
                pc_len=pl.col('pc').fill_null('').str.len_chars().filter(pl.col('pc').fill_null('') != '').mean(),
                core_ntok=pl.col('core').fill_null('').str.split(' ').list.len().mean(),
            ).sort('country')
            print(f'== {sp} S{src}')
            print(r)

    # sibling coverage under the current threshold (sib_thr=0.9), from the b1ka stage-1 scores, one-to-one margin .2
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from decide import one2one  # noqa: E402
    for sp, pte in (('tsim', 'oof_b1ka'), ('test', 'pte_b1ka')):
        p = pl.read_parquet(f'{W}/{pte}.parquet')
        s1c = pl.read_parquet(f'{W}/n_{sp}_s1.parquet', columns=['id', 'country']).select(enc('id').alias('id1'), 'country')
        for thr in (0.9, 0.7, 0.5):
            sib = one2one(p, 0.2).filter(pl.col('p') >= thr).group_by('id1').len('nsib')
            r = s1c.join(sib, on='id1', how='left').with_columns(pl.col('nsib').fill_null(0)).group_by('country').agg(
                mean_sib=pl.col('nsib').mean(), zero_sib=(pl.col('nsib') == 0).mean())
            print(f'== {sp} sib_thr={thr}')
            print(r.sort('country'))
