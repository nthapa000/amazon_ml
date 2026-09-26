"""LLM judge: zero-shot "same business?" score for the pairs the final model is unsure about.
Qwen/Qwen2.5-7B-Instruct (Apache-2.0, 7.6B params), bf16, one forward pass per pair (no generation, no training):
score = logit("Yes") - logit("No") at the answer position. Raw names and addresses of the two records only.

python llm.py split=tsim pred=oof_fin2b lo=0.1 hi=0.9 frac=0.5 shard=0 nshard=2 out=llm_tsim [lim=2000 bs=32]
-> W/{out}_s{shard}.parquet  (id1, id2, llm)
"""
import os
import sys
import time

import numpy as np
import polars as pl
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(__file__))
from block import enc  # noqa: E402

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')
SYS = 'You compare business records that come from different databases.'
ASK = ('Do record A and record B refer to the same business (the same company at the same location)? Records of the '
       'same business may differ by typos, abbreviations, missing or reordered address parts, accents, legal forms or '
       'trade names; different businesses may share a word, a street or a city. Answer with Yes or No.')


def texts(split):
    t = []
    for k in (1, 2, 3):
        d = pl.read_parquet(f'{W}/n_{split}_s{k}.parquet', columns=['id', 'name', 'addr'])
        t.append(d.select(enc('id').alias('id'), pl.col('name').str.slice(0, 160).alias('n'),
                          pl.col('addr').str.slice(0, 160).alias('a')))
    return pl.concat(t)


def prompt(tok, n1, a1, n2, a2):
    u = f'Record A\nname: {n1}\naddress: {a1 or "(none)"}\n\nRecord B\nname: {n2}\naddress: {a2 or "(none)"}\n\n{ASK}'
    return tok.apply_chat_template([{'role': 'system', 'content': SYS}, {'role': 'user', 'content': u}],
                                   tokenize=False, add_generation_prompt=True)


def main(kv):
    t0 = time.time()
    sp, sh, ns = kv['split'], int(kv.get('shard', 0)), int(kv.get('nshard', 1))
    lo, hi = float(kv.get('lo', 0.1)), float(kv.get('hi', 0.9))
    d = pl.read_parquet(f'{W}/{kv["pred"]}.parquet').filter((pl.col('p') > lo) & (pl.col('p') < hi)).select('id1', 'id2')
    if 'frac' in kv:
        d = d.filter(pl.col('id1').hash(seed=23) % 1000 < int(float(kv['frac']) * 1000))
    d = d.filter(pl.col('id1').hash(seed=29) % ns == sh)
    if 'lim' in kv:
        d = d.head(int(kv['lim']))
    tx = texts(sp)
    d = (d.join(tx.rename({'id': 'id1', 'n': 'n1', 'a': 'a1'}), on='id1', how='left')
         .join(tx.rename({'id': 'id2', 'n': 'n2', 'a': 'a2'}), on='id2', how='left').fill_null(''))
    del tx
    mn = kv.get('model', 'Qwen/Qwen2.5-7B-Instruct')
    tok = AutoTokenizer.from_pretrained(mn)
    tok.padding_side = 'left'
    yes, no = tok.encode('Yes', add_special_tokens=False)[0], tok.encode('No', add_special_tokens=False)[0]
    model = AutoModelForCausalLM.from_pretrained(mn, dtype=torch.bfloat16).to('cuda').eval()
    ps = [prompt(tok, *r) for r in zip(d['n1'], d['a1'], d['n2'], d['a2'])]
    order = np.argsort([len(p) for p in ps])                 # similar lengths per batch -> little padding
    out = np.zeros(len(ps), dtype=np.float32)
    bs = int(kv.get('bs', 32))
    print(f'{sp} shard {sh}/{ns}: {len(ps)} pairs, model loaded {time.time() - t0:.0f}s', flush=True)
    t1 = time.time()
    with torch.inference_mode():
        for i in range(0, len(ps), bs):
            ix = order[i:i + bs]
            e = tok([ps[j] for j in ix], return_tensors='pt', padding=True).to('cuda')
            h = model.model(**e).last_hidden_state[:, -1]          # answer position only (saves the vocab logits)
            lg = model.lm_head(h).float()
            out[ix] = (lg[:, yes] - lg[:, no]).cpu().numpy()
            if (i // bs) % 200 == 0:
                print(f'  {i + len(ix)}/{len(ps)} {(i + len(ix)) / (time.time() - t1):.1f} pairs/s', flush=True)
    r = d.select('id1', 'id2').with_columns(llm=pl.Series(out))
    r.write_parquet(f'{W}/{kv["out"]}_s{sh}.parquet')
    print(f'wrote {kv["out"]}_s{sh} {r.height} pairs, {time.time() - t0:.0f}s, mean llm {out.mean():.3f}', flush=True)


if __name__ == '__main__':
    main(dict(a.split('=', 1) for a in sys.argv[1:]))
