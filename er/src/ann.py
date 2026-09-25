"""ANN recall probe / blocking pass (GPU): nearest S1 neighbours of every S2/S3 record by embedding similarity.

Embeds normalized "core name | address" with a small multilingual encoder (mean pooling, L2-normalized), then for
each S2/S3 record takes the top-k S1 of the same country by cosine similarity. New pairs = not in the base
candidate set. With labels (tsim) it reports how many of the base-missed true links are recovered.

python ann.py split=tsim base=cand_tsim_kx k=10 model=intfloat/multilingual-e5-small out=ann_tsim
"""
import os
import sys
import time

import numpy as np
import polars as pl
import torch
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, os.path.dirname(__file__))
from block import enc, gt_path  # noqa: E402

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')
DEV = 'cuda'


def load(split):
    fr = []
    for k in (1, 2, 3):
        d = pl.read_parquet(f'{W}/n_{split}_s{k}.parquet', columns=['id', 'country', 'core', 'ad'])
        fr.append(d.with_columns(side=pl.lit(int(k > 1), pl.Int8)))
    d = pl.concat(fr).with_columns(id=enc('id'))
    return d.with_columns(t=('query: ' + pl.col('core') + ' | ' + pl.col('ad')).str.slice(0, 200))


@torch.no_grad()
def embed(texts, tok, model, bs=1024):
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float16)
    for i in range(0, len(texts), bs):
        e = tok(texts[i:i + bs], truncation=True, max_length=64, padding=True, return_tensors='pt').to(DEV)
        with torch.autocast('cuda', dtype=torch.float16):
            h = model(**e).last_hidden_state
        m = e['attention_mask'].unsqueeze(-1).to(h.dtype)
        v = (h * m).sum(1) / m.sum(1)
        out[i:i + bs] = torch.nn.functional.normalize(v.float(), dim=-1).half().cpu().numpy()
    return out


def main(kv):
    t0 = time.time()
    split, k = kv['split'], int(kv.get('k', 10))
    tok = AutoTokenizer.from_pretrained(kv.get('model', 'intfloat/multilingual-e5-small'))
    model = AutoModel.from_pretrained(kv.get('model', 'intfloat/multilingual-e5-small')).to(DEV).eval()
    rec = load(split)
    res = []
    for c in rec['country'].unique().sort().to_list():
        r = rec.filter(pl.col('country') == c)
        a, b = r.filter(pl.col('side') == 0), r.filter(pl.col('side') == 1)
        ea = torch.from_numpy(embed(a['t'].to_list(), tok, model)).to(DEV)
        eb = embed(b['t'].to_list(), tok, model)
        print(c, 'embedded', a.height, b.height, f'{time.time() - t0:.0f}s', flush=True)
        ids_a = a['id'].to_numpy()
        top_i, top_s = [], []
        for i in range(0, len(eb), 4096):
            s = torch.from_numpy(eb[i:i + 4096]).to(DEV) @ ea.T
            v, j = s.topk(k, dim=1)
            top_i.append(j.cpu().numpy())
            top_s.append(v.float().cpu().numpy())
        ti, ts = np.vstack(top_i), np.vstack(top_s)
        res.append(pl.DataFrame({'id2': np.repeat(b['id'].to_numpy(), k), 'id1': ids_a[ti.ravel()],
                                 'ann_s': ts.ravel(), 'ann_r': np.tile(np.arange(1, k + 1, dtype=np.int16), len(b))}))
        del ea
        torch.cuda.empty_cache()
        print(c, 'searched', f'{time.time() - t0:.0f}s', flush=True)
    nn = pl.concat(res)
    base = pl.read_parquet(f'{W}/{kv["base"]}.parquet', columns=['id1', 'id2'])
    new = nn.join(base, on=['id1', 'id2'], how='anti')
    new.write_parquet(f'{W}/{kv.get("out", "ann_" + split)}.parquet')
    print('ann pairs', nn.height, 'new (not in base)', new.height, flush=True)
    if split != 'test':
        gt = pl.read_parquet(gt_path(split)).select(enc('s1').alias('id1'), enc('id2').alias('id2'))
        miss = gt.join(base, on=['id1', 'id2'], how='anti')
        for kk in (1, 3, 5, 10, k):
            nk = new.filter(pl.col('ann_r') <= kk)
            rec_n = miss.join(nk, on=['id1', 'id2'], how='semi').height
            print(f'top-{kk}: recovers {rec_n} of {miss.height} base-missed true links ({rec_n / miss.height:.3f}); '
                  f'new pairs {nk.height} ({nk.height / max(base["id1"].n_unique(), 1):.1f} per S1)', flush=True)


if __name__ == '__main__':
    main(dict(a.split('=', 1) for a in sys.argv[1:]))
