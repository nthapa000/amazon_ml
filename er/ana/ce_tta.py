"""Text-view test-time augmentation for a saved cross-encoder on unseen-country test pairs (plan (1).md N4). No src/ edits.

S2/S3 inject spurious accents and upper-case names, S1 does not. Pairs whose S2/S3 raw text carries that noise
(any non-ASCII character, or an all-caps name) are re-scored on a folded view of BOTH records (unidecode + lowercase);
their score becomes the mean of the raw-view and folded-view logits. All other rows keep the model's own score.

  GPU: python ana/ce_tta.py tag=b1ka model=ceqf half=0|1 [lo=0.003 hi=0.997 ml=192 pbs=128]
       -> W/<model>t_<tag>_test_h<half>.parquet (id1, id2, fold)
  CPU: python ana/ce_tta.py merge=1 tag=b1ka model=ceqf
       -> W/<model>t_<tag>_test.parquet (id1, id2, <model>t) = W/<model>_<tag>_test.parquet with the TTA rows averaged
"""
import os
import sys
import json
import time

import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from block import enc  # noqa: E402

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')
kv = dict(a.split('=', 1) for a in sys.argv[1:])
M, TAG = kv.get('model', 'ceqf'), kv.get('tag', 'b1ka')
NOISE = r'[^\x00-\x7F]'


def records():
    return pl.concat([pl.read_parquet(f'{W}/n_test_s{k}.parquet', columns=['id', 'name', 'addr']) for k in (1, 2, 3)]).with_columns(
        id=enc('id'), name=pl.col('name').fill_null(''), addr=pl.col('addr').fill_null(''))


def view(d, fold):
    if fold:
        from unidecode import unidecode
        f = lambda s: unidecode(s).lower()                      # noqa: E731
        d = d.with_columns(pl.col('name').map_elements(f, return_dtype=pl.String),
                           pl.col('addr').map_elements(f, return_dtype=pl.String))
    t = ('name: ' + pl.col('name') + ' | address: ' + pl.when(pl.col('addr') != '').then(pl.col('addr')).otherwise(pl.lit('none')))
    return d.select('id', t.str.slice(0, 200).alias('t'))       # same format as ce.texts(mode='raw')


def gpu():
    import torch
    import ce
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import PeftModel
    ce.AMP, ce.ML = torch.bfloat16, int(kv.get('ml', 192))
    h, t0 = int(kv['half']), time.time()
    lo, hi = float(kv.get('lo', 0.003)), float(kv.get('hi', 0.997))
    te = pl.read_parquet(f'{W}/pte_{TAG}.parquet').filter((pl.col('p') > lo) & (pl.col('p') < hi))
    tb = ce.keep_unseen(te).select('id1', 'id2')
    rec = records()
    noisy = rec.filter(pl.col('name').str.contains(NOISE) | pl.col('addr').str.contains(NOISE)
                       | ((pl.col('name') == pl.col('name').str.to_uppercase()) & pl.col('name').str.contains('[A-Za-z]{2}')))
    tb = tb.join(noisy.select(pl.col('id').alias('id2')), on='id2', how='semi')
    need = pl.concat([tb.select(pl.col('id1').alias('id')), tb.select(pl.col('id2').alias('id'))]).unique()
    tx = view(rec.join(need, on='id', how='semi'), fold=True)
    d = ce.attach(tb, tx)
    print(f'half {h}: unseen band pairs with noisy S2/S3 text {d.height} ({time.time() - t0:.0f}s)', flush=True)
    md = f"{os.environ.get('ER_MODELS', W + '/models')}/{M}_{TAG}_h{h}"
    base = json.load(open(f'{md}/adapter_config.json'))['base_model_name_or_path']
    tok = AutoTokenizer.from_pretrained(md)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=1, ignore_mismatched_sizes=True,
                                                               dtype=torch.bfloat16)
    model.config.pad_token_id = tok.pad_token_id
    model = PeftModel.from_pretrained(model, md).to(ce.DEV)
    s = ce.predict(model, tok, d, int(kv.get('pbs', 128)))
    d.select('id1', 'id2').with_columns(fold=pl.Series(s)).write_parquet(f'{W}/{M}t_{TAG}_test_h{h}.parquet')
    print(f'half {h} folded view scored {d.height} {time.time() - t0:.0f}s', flush=True)


def merge():
    b = pl.read_parquet(f'{W}/{M}_{TAG}_test.parquet')
    f = pl.concat([pl.read_parquet(f'{W}/{M}t_{TAG}_test_h{h}.parquet') for h in (0, 1)]).group_by('id1', 'id2').agg(
        pl.col('fold').mean(), pl.len().alias('n'))
    assert f['n'].min() == 2, 'both halves must score every TTA pair'
    x = b.join(f.select('id1', 'id2', 'fold'), on=['id1', 'id2'], how='left')
    assert x.filter(pl.col('fold').is_not_null()).height == f.height, 'TTA pairs missing from the base file'
    out = x.select('id1', 'id2', pl.when(pl.col('fold').is_null()).then(pl.col(M)).otherwise((pl.col(M) + pl.col('fold')) / 2)
                   .alias(f'{M}t'))
    out.write_parquet(f'{W}/{M}t_{TAG}_test.parquet')
    ch = x.filter(pl.col('fold').is_not_null())
    import numpy as np
    from scipy.stats import spearmanr
    a, c = ch[M].to_numpy(), ch['fold'].to_numpy()
    print(f'merged: {ch.height} of {b.height} test rows averaged; raw vs folded spearman {spearmanr(a, c).correlation:.4f}, '
          f'mean raw {a.mean():+.3f} folded {c.mean():+.3f}, sign flips {np.mean((a > 0) != (c > 0)):.4f}', flush=True)


if __name__ == '__main__':
    merge() if kv.get('merge') == '1' else gpu()
