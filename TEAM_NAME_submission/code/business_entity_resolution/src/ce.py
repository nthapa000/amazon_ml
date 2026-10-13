"""Cross-encoder re-scorer for the uncertain band of stage-1 scores (GPU).

Two-fold by S1 id (hash % 2): a model trained on half A scores band pairs of half B (OOF), and both
models score the test band (averaged). Output feature `ce` (logit) for band pairs, NaN elsewhere.

python ce.py split=tsim tag=b1ka lo=0.003 hi=0.997 ntr=3000000 epochs=2 txt=raw model=<model dir> name=ceb
  [half=h (one half per process, then merge=1)] [init=ceb_b1ka (warm start from W/models/<init>_h<h>)]
  [extra=fra_b1ka (added training pairs)] [lora=16 bf16=1 ml=192 (decoder rerankers)] [save=0 (do not keep weights)]
-> W/{name}_{tag}_train.parquet (OOF band scores), W/{name}_{tag}_test.parquet  (id1, id2, {name})
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

W = os.environ.get('ER_WORK', 'work')
DEV = 'cuda'
AMP, ML = torch.float16, 128          # bf16=1 for decoder LLM rerankers (fp16 overflows); ml= max tokens per pair


def texts(split, mode='norm'):
    """Text of every record of a split: raw 'name: .. | address: ..' (script and case kept) or normalized."""
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
    """Join the texts of both records to a pair list."""
    return (pairs.join(tx.rename({'id': 'id1', 't': 't1'}), on='id1', how='left')
            .join(tx.rename({'id': 'id2', 't': 't2'}), on='id2', how='left')
            .with_columns(pl.col('t1').fill_null(''), pl.col('t2').fill_null('')))


class Coll:
    """Batch collator: tokenizes (record 1, record 2) as a text pair, or as one two-line text for decoders."""
    def __init__(self, tok, ml=128):
        """tok: tokenizer; ml: maximum number of tokens per pair."""
        self.tok, self.ml = tok, ml

    def __call__(self, b):
        """List of (text 1, text 2, label) -> model inputs with float labels."""
        a, c, y = zip(*b)
        if self.tok.sep_token is None:                  # decoder tokenizers (Qwen): one sequence, records on 2 lines
            e = self.tok([f'{x}\n{z}' for x, z in zip(a, c)], truncation=True, max_length=self.ml, padding=True,
                         return_tensors='pt')
        else:
            e = self.tok(list(a), list(c), truncation=True, max_length=self.ml, padding=True, return_tensors='pt')
        e['labels'] = torch.tensor(y, dtype=torch.float32)
        return e


def predict(model, tok, d, bs=1024):
    """Logit score of every pair in d (batched, mixed precision)."""
    model.eval()
    ds = list(zip(d['t1'].to_list(), d['t2'].to_list(), [0.0] * d.height))
    dl = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=Coll(tok, ML), num_workers=6)
    out = []
    with torch.no_grad(), torch.autocast('cuda', dtype=AMP):
        for e in dl:
            e.pop('labels')
            out.append(model(**{k: v.to(DEV, non_blocking=True) for k, v in e.items()}).logits.float().squeeze(-1).cpu())
    return torch.cat(out).numpy()


def train(mname, d, epochs=1, bs=256, lr=4e-5, lora=0):
    """Fine-tune a one-logit sequence classifier (BCE, AdamW, one-cycle schedule); optional LoRA."""
    tok = AutoTokenizer.from_pretrained(mname)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(dtype=torch.bfloat16) if AMP == torch.bfloat16 else {}
    model = AutoModelForSequenceClassification.from_pretrained(mname, num_labels=1, ignore_mismatched_sizes=True, **kw)
    model.config.pad_token_id = tok.pad_token_id
    if lora:                                             # LoRA on all linear layers + trainable new score head
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=lora, lora_alpha=2 * lora, lora_dropout=0.05, target_modules='all-linear',
                                                 modules_to_save=['score'], task_type='SEQ_CLS'))
        model.print_trainable_parameters()
    model = model.to(DEV)
    ds = list(zip(d['t1'].to_list(), d['t2'].to_list(), d['y'].cast(pl.Float32).to_list()))
    dl = DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=Coll(tok, ML), num_workers=6, drop_last=True)
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=lr, weight_decay=0.01)
    tot = epochs * len(dl)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=tot, pct_start=0.05)
    scaler = torch.amp.GradScaler('cuda', enabled=AMP == torch.float16)
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    t0, st = time.time(), 0
    for _ in range(epochs):
        for e in dl:
            y = e.pop('labels').to(DEV)
            with torch.autocast('cuda', dtype=AMP):
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
    """Per half of the S1 ids: train on the other half, score this half's band (OOF) and the test band."""
    global AMP, ML
    AMP = torch.bfloat16 if kv.get('bf16') == '1' else torch.float16
    ML = int(kv.get('ml', 128))
    split, tag = kv.get('split', 'tsim'), kv['tag']
    lo, hi = float(kv.get('lo', 0.02)), float(kv.get('hi', 0.98))
    slo, shi = float(kv.get('slo', lo)), float(kv.get('shi', hi))      # scoring band (default = training band)
    ntr = int(kv.get('ntr', 2_000_000))
    mname = kv['model']
    o = pl.read_parquet(f'{W}/oof_{tag}.parquet').with_columns(h=(pl.col('id1').hash(seed=5) % 2).cast(pl.Int8))
    te = pl.read_parquet(f'{W}/pte_{tag}.parquet')
    band = o.filter((pl.col('p') > slo) & (pl.col('p') < shi))
    tband = te.filter((pl.col('p') > slo) & (pl.col('p') < shi))
    if 'lim' in kv:
        band, tband = band.sample(int(kv['lim']), seed=0), tband.head(int(kv['lim']))
    print('band train', band.height, 'pos', band['y'].sum(), 'band test', tband.height, flush=True)
    mode = kv.get('txt', 'norm')
    tx = texts(split, mode)
    txt = texts('test', mode)
    tband = attach(tband.select('id1', 'id2'), txt)
    del txt
    res, pt = [], np.zeros(tband.height, dtype=np.float32)
    hs = (int(kv['half']),) if 'half' in kv else (0, 1)  # half=h: one half per job (2 GPUs in parallel), then merge=1
    for h in hs:
        t0 = time.time()
        src = o.filter(pl.col('h') != h)
        b = src.filter((pl.col('p') > lo) & (pl.col('p') < hi))
        b = b.sample(min(b.height, int(ntr * 0.85)), seed=h)
        e = src.filter((pl.col('p') <= lo) | (pl.col('p') >= hi))
        e = e.sample(min(e.height, int(ntr * 0.15)), seed=h)
        trd = attach(pl.concat([b, e]).select('id1', 'id2', 'y'), tx).select('t1', 't2', pl.col('y').cast(pl.Float32))
        if 'extra' in kv:                                  # extra (t1, t2, y) pairs, e.g. self-training set (fradapt.py)
            ex = pl.read_parquet(f"{W}/{kv['extra']}.parquet").select('t1', 't2', pl.col('y').cast(pl.Float32))
            trd = pl.concat([trd, ex.head(int(kv['nx'])) if 'nx' in kv else ex])
        mi = f"{os.environ.get('ER_MODELS', W + '/models')}/{kv['init']}_h{h}" if 'init' in kv else mname
        print(f'half {h}: train pairs {trd.height}, init {mi}', flush=True)  # init: warm start from saved half model
        model, tok = train(mi, trd, epochs=int(kv.get('epochs', 1)), bs=int(kv.get('bs', 256)),
                           lr=float(kv.get('lr', 4e-5)), lora=int(kv.get('lora', 0)))
        print(f'half {h} trained {time.time() - t0:.0f}s', flush=True)
        if kv.get('save', '1') == '1':                     # keep weights: test pairs can be re-scored later
            md = f"{os.environ.get('ER_MODELS', W + '/models')}/{kv.get('name', 'ce')}_{tag}_h{h}"
            model.save_pretrained(md)
            tok.save_pretrained(md)
        vb = attach(band.filter(pl.col('h') == h).select('id1', 'id2'), tx)
        vb = vb.with_columns(ce=pl.Series(predict(model, tok, vb, int(kv.get('pbs', 1024)))))
        res.append(vb.select('id1', 'id2', 'ce'))
        print(f'half {h} oof scored {vb.height} {time.time() - t0:.0f}s', flush=True)
        pt += predict(model, tok, tband, int(kv.get('pbs', 1024))) / len(hs)
        print(f'half {h} test scored {time.time() - t0:.0f}s', flush=True)
        del model
        torch.cuda.empty_cache()
    oo = pl.concat(res)
    nm = kv.get('name', 'ce')
    sfx = f'_h{hs[0]}' if len(hs) == 1 else ''
    oo.rename({'ce': nm}).write_parquet(f'{W}/{nm}_{tag}_train{sfx}.parquet')
    tband.select('id1', 'id2').with_columns(pl.Series(nm, pt)).write_parquet(f'{W}/{nm}_{tag}_test{sfx}.parquet')
    # quick diagnostic: AUC of ce vs p inside the band
    from sklearn.metrics import roc_auc_score
    chk = band.join(oo, on=['id1', 'id2'])
    print('band AUC p', roc_auc_score(chk['y'], chk['p']), 'ce', roc_auc_score(chk['y'], chk['ce']), flush=True)


def merge(kv):
    """combine the two half jobs: OOF band scores concatenated, test scores averaged."""
    nm, tag = kv['name'], kv['tag']
    tr = pl.concat([pl.read_parquet(f'{W}/{nm}_{tag}_train_h{h}.parquet') for h in (0, 1)])
    te = [pl.read_parquet(f'{W}/{nm}_{tag}_test_h{h}.parquet') for h in (0, 1)]
    te = te[0].join(te[1].rename({nm: 'b'}), on=['id1', 'id2']).with_columns(((pl.col(nm) + pl.col('b')) / 2).alias(nm)).drop('b')
    tr.write_parquet(f'{W}/{nm}_{tag}_train.parquet')
    te.write_parquet(f'{W}/{nm}_{tag}_test.parquet')
    from sklearn.metrics import roc_auc_score
    chk = pl.read_parquet(f'{W}/oof_{tag}.parquet').join(tr, on=['id1', 'id2'])
    print('merged', tr.height, te.height, 'band AUC p', roc_auc_score(chk['y'], chk['p']), 'ce', roc_auc_score(chk['y'], chk[nm]), flush=True)


if __name__ == '__main__':
    kv = dict(a.split('=', 1) for a in sys.argv[1:])
    merge(kv) if kv.get('merge') == '1' else main(kv)
