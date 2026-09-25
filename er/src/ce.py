"""Cross-encoder re-scorer for the uncertain band of stage-1 scores (GPU).

Two-fold by S1 id (hash % 2): a model trained on half A scores band pairs of half B (OOF), and both
models score the test band (averaged). Output feature `ce` (logit) for band pairs, NaN elsewhere.

python ce.py split=tsim tag=t1 lo=0.02 hi=0.98 ntr=2000000 model=cross-encoder/ms-marco-MiniLM-L6-v2
-> W/ce_{tag}_train.parquet, W/ce_{tag}_test.parquet  (id1, id2, ce)
"""
import os
import sys
import time

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, os.path.dirname(__file__))
from block import enc  # noqa: E402

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')
DEV = 'cuda'


def texts(split, mode='norm'):
    if mode == 'raw':                                # original text: native scripts, case, punctuation kept
        fr = [pl.read_parquet(f'{W}/n_{split}_s{k}.parquet', columns=['id', 'name', 'addr']) for k in (1, 2, 3)]
        d = pl.concat(fr).with_columns(id=enc('id')).with_columns(pl.col('name').fill_null(''),
                                                                   pl.col('addr').fill_null(''))
        t = ('name: ' + pl.col('name') + ' | address: '
             + pl.when(pl.col('addr') != '').then(pl.col('addr')).otherwise(pl.lit('none')))
        return d.select('id', t.str.slice(0, 200).alias('t'))
    c = ['id', 'nm', 'alt', 'dom', 'ad']
    fr = [pl.read_parquet(f'{W}/n_{split}_s{k}.parquet', columns=c) for k in (1, 2, 3)]
    d = pl.concat(fr).with_columns(id=enc('id'))
    t = (pl.col('nm') + pl.when(pl.col('alt') != '').then(' aka ' + pl.col('alt')).otherwise(pl.lit(''))
         + pl.when(pl.col('dom') != '').then(' www ' + pl.col('dom')).otherwise(pl.lit(''))
         + ' | ' + pl.when(pl.col('ad') != '').then(pl.col('ad')).otherwise(pl.lit('no address')))
    return d.select('id', t.str.slice(0, 160).alias('t'))


def attach(pairs, tx):
    return (pairs.join(tx.rename({'id': 'id1', 't': 't1'}), on='id1', how='left')
            .join(tx.rename({'id': 'id2', 't': 't2'}), on='id2', how='left')
            .with_columns(pl.col('t1').fill_null(''), pl.col('t2').fill_null('')))


class Coll:
    def __init__(self, tok, ml=128):
        self.tok, self.ml = tok, ml

    def __call__(self, b):
        a, c, y = zip(*b)
        e = self.tok(list(a), list(c), truncation=True, max_length=self.ml, padding=True, return_tensors='pt')
        e['labels'] = torch.tensor(y, dtype=torch.float32)
        return e


def predict(model, tok, d, bs=1024):
    model.eval()
    ds = list(zip(d['t1'].to_list(), d['t2'].to_list(), [0.0] * d.height))
    dl = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=Coll(tok), num_workers=6)
    out = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        for e in dl:
            e.pop('labels')
            out.append(model(**{k: v.to(DEV, non_blocking=True) for k, v in e.items()}).logits.float().squeeze(-1).cpu())
    return torch.cat(out).numpy()


def train(mname, d, epochs=1, bs=256, lr=4e-5):
    tok = AutoTokenizer.from_pretrained(mname)
    model = AutoModelForSequenceClassification.from_pretrained(mname, num_labels=1,
                                                               ignore_mismatched_sizes=True).to(DEV)
    ds = list(zip(d['t1'].to_list(), d['t2'].to_list(), d['y'].cast(pl.Float32).to_list()))
    dl = DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=Coll(tok), num_workers=6, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    tot = epochs * len(dl)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=tot, pct_start=0.05)
    scaler = torch.cuda.amp.GradScaler()
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    t0, st = time.time(), 0
    for _ in range(epochs):
        for e in dl:
            y = e.pop('labels').to(DEV)
            with torch.autocast('cuda', dtype=torch.float16):
                lo = model(**{k: v.to(DEV, non_blocking=True) for k, v in e.items()}).logits.squeeze(-1)
            loss = lossf(lo.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sch.step()
            st += 1
            if st % 500 == 0:
                print(f'  step {st}/{tot} loss {loss.item():.4f} {time.time() - t0:.0f}s', flush=True)
    return model, tok


def main(kv):
    split, tag = kv.get('split', 'tsim'), kv['tag']
    lo, hi = float(kv.get('lo', 0.02)), float(kv.get('hi', 0.98))
    ntr = int(kv.get('ntr', 2_000_000))
    mname = kv.get('model', 'cross-encoder/ms-marco-MiniLM-L6-v2')
    o = pl.read_parquet(f'{W}/oof_{tag}.parquet').with_columns(h=(pl.col('id1').hash(seed=5) % 2).cast(pl.Int8))
    te = pl.read_parquet(f'{W}/pte_{tag}.parquet')
    band = o.filter((pl.col('p') > lo) & (pl.col('p') < hi))
    tband = te.filter((pl.col('p') > lo) & (pl.col('p') < hi))
    if 'lim' in kv:
        band, tband = band.sample(int(kv['lim']), seed=0), tband.head(int(kv['lim']))
    print('band train', band.height, 'pos', band['y'].sum(), 'band test', tband.height, flush=True)
    mode = kv.get('txt', 'norm')
    tx = texts(split, mode)
    txt = texts('test', mode)
    tband = attach(tband.select('id1', 'id2'), txt)
    del txt
    res, pt = [], np.zeros(tband.height, dtype=np.float32)
    for h in (0, 1):
        t0 = time.time()
        src = o.filter(pl.col('h') != h)
        b = src.filter((pl.col('p') > lo) & (pl.col('p') < hi))
        b = b.sample(min(b.height, int(ntr * 0.85)), seed=h)
        e = src.filter((pl.col('p') <= lo) | (pl.col('p') >= hi))
        e = e.sample(min(e.height, int(ntr * 0.15)), seed=h)
        trd = attach(pl.concat([b, e]).select('id1', 'id2', 'y'), tx)
        model, tok = train(mname, trd, epochs=int(kv.get('epochs', 1)), bs=int(kv.get('bs', 256)),
                           lr=float(kv.get('lr', 4e-5)))
        print(f'half {h} trained {time.time() - t0:.0f}s', flush=True)
        md = f"{os.environ.get('ER_MODELS', W + '/models')}/{kv.get('name', 'ce')}_{tag}_h{h}"
        model.save_pretrained(md)                          # keep weights: test pairs can be re-scored later
        tok.save_pretrained(md)
        vb = attach(band.filter(pl.col('h') == h).select('id1', 'id2'), tx)
        vb = vb.with_columns(ce=pl.Series(predict(model, tok, vb, int(kv.get('pbs', 1024)))))
        res.append(vb.select('id1', 'id2', 'ce'))
        print(f'half {h} oof scored {vb.height} {time.time() - t0:.0f}s', flush=True)
        pt += predict(model, tok, tband, int(kv.get('pbs', 1024))) / 2
        print(f'half {h} test scored {time.time() - t0:.0f}s', flush=True)
        del model
        torch.cuda.empty_cache()
    oo = pl.concat(res)
    nm = kv.get('name', 'ce')
    oo.rename({'ce': nm}).write_parquet(f'{W}/{nm}_{tag}_train.parquet')
    tband.select('id1', 'id2').with_columns(pl.Series(nm, pt)).write_parquet(f'{W}/{nm}_{tag}_test.parquet')
    # quick diagnostic: AUC of ce vs p inside the band
    from sklearn.metrics import roc_auc_score
    chk = band.join(oo, on=['id1', 'id2'])
    print('band AUC p', roc_auc_score(chk['y'], chk['p']), 'ce', roc_auc_score(chk['y'], chk['ce']), flush=True)


if __name__ == '__main__':
    main(dict(a.split('=', 1) for a in sys.argv[1:]))
