"""TEMPLATE: end-to-end multimodal regressor (text + image + tabular), in the style of the
2025 Rank 8 (CLIP + DistilBERT, Huber on log target, contrastive aux) and Rank 20 (SigLIP,
cross-attention fusion, small FFN head, SMAPE loss) solutions.

NOT executed in the skill's build environment (no GPU/torch there). Expect to adjust model names,
batch size and image paths. Keep its outputs in the same oof_<tag>.npy / tst_<tag>.npy format so
blend.py can combine it with the GBDTs.

python mm_finetune.py train=train.csv test=test.csv target=price id=sample_id folds=folds.csv \
    text=catalog_content img_dir=images feats=f_tr.csv:f_te.csv tag=mm1 fold=-1
    fold=-1 runs all folds; fold=k runs one fold (split folds across GPUs / teammates).

Before you pick encoders: verify licences and parameter caps against the challenge rules.
"""
import math
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
from kv import args, cols, read

a = args({'train': 'train.csv', 'test': 'test.csv', 'target': 'target', 'id': 'sample_id',
          'folds': 'folds.csv', 'text': 'catalog_content', 'img_dir': 'images', 'feats': '',
          'txt_model': 'google/embeddinggemma-300m', 'img_model': 'google/siglip2-base-patch16-224',
          'max_len': 256, 'bs': 32, 'ep': 8, 'lr': 2e-5, 'head_lr': 1e-3, 'wd': 0.01, 'warm': 0.06,
          'aux': 0.2, 'smape_from': 3, 'fold': -1, 'tag': 'mm', 'seed': 42, 'workers': 4})
torch.manual_seed(a['seed']); np.random.seed(a['seed'])
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
tr, te = read(a['train']), read(a['test'])
tab_cols = []
for pair in cols(a['feats']):
    ftr, fte = pair.split(':')
    f1, f2 = read(ftr), read(fte)
    num = [c for c in f1.columns if c != a['id'] and pd.api.types.is_numeric_dtype(f1[c])]
    tr = tr.merge(f1[[a['id']] + num], on=a['id'], how='left')
    te = te.merge(f2[[a['id']] + num], on=a['id'], how='left')
    tab_cols += num
fo = read(a['folds'])
fold = tr[a['id']].map(dict(zip(fo[fo.columns[0]], fo['fold']))).values.astype(int)
y = np.log(tr[a['target']].values.astype(float))
mu, sd = y.mean(), y.std()
# tabular: log1p + standardise with train stats, NaN -> 0 plus the model sees has-image flag
T = pd.concat([tr[tab_cols], te[tab_cols]]).astype(float)
T = np.sign(T) * np.log1p(T.abs())
T = ((T - T.iloc[:len(tr)].mean()) / (T.iloc[:len(tr)].std() + 1e-6)).fillna(0).values.astype('float32')
Ttr, Tte = T[:len(tr)], T[len(tr):]

tok = AutoTokenizer.from_pretrained(a['txt_model'])
proc = AutoImageProcessor.from_pretrained(a['img_model'])


class DS(torch.utils.data.Dataset):
    def __init__(s, df, tab, yy=None):
        s.t = df[a['text']].fillna('').astype(str).tolist()
        s.i = df[a['id']].tolist(); s.tab = tab; s.y = yy

    def __len__(s):
        return len(s.t)

    def __getitem__(s, k):
        p = f"{a['img_dir']}/{s.i[k]}.jpg"
        ok = os.path.exists(p)
        try:
            im = Image.open(p).convert('RGB') if ok else Image.new('RGB', (224, 224))
        except Exception:
            im, ok = Image.new('RGB', (224, 224)), False
        px = proc(images=im, return_tensors='pt')['pixel_values'][0]
        r = {'txt': s.t[k], 'px': px, 'has': float(ok), 'tab': s.tab[k]}
        if s.y is not None:
            r['y'] = (s.y[k] - mu) / sd
        return r


def collate(b):
    e = tok([x['txt'] for x in b], padding=True, truncation=True, max_length=a['max_len'], return_tensors='pt')
    o = {'ids': e['input_ids'], 'am': e['attention_mask'], 'px': torch.stack([x['px'] for x in b]),
         'has': torch.tensor([x['has'] for x in b]).unsqueeze(1), 'tab': torch.tensor(np.stack([x['tab'] for x in b]))}
    if 'y' in b[0]:
        o['y'] = torch.tensor([x['y'] for x in b], dtype=torch.float32)
    return o


class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.te = AutoModel.from_pretrained(a['txt_model'])
        vm = AutoModel.from_pretrained(a['img_model'])
        s.ve = vm.vision_model if hasattr(vm, 'vision_model') else vm
        dt, dv, d = s.te.config.hidden_size, s.ve.config.hidden_size, 512
        s.pt, s.pv = nn.Linear(dt, d), nn.Linear(dv, d)
        s.t2i = nn.MultiheadAttention(d, 8, batch_first=True)      # text queries image
        s.i2t = nn.MultiheadAttention(d, 8, batch_first=True)      # image queries text
        s.ln = nn.LayerNorm(d)
        nt = len(tab_cols) + 1
        s.head = nn.Sequential(nn.Linear(2 * d + nt, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(.1),
                               nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(),
                               nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1))
        s.tau = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def forward(s, b):
        ht = s.pt(s.te(input_ids=b['ids'], attention_mask=b['am']).last_hidden_state)
        hv = s.pv(s.ve(pixel_values=b['px']).last_hidden_state) * b['has'].unsqueeze(-1)
        a1, _ = s.t2i(ht, hv, hv)
        a2, _ = s.i2t(hv, ht, ht, key_padding_mask=b['am'] == 0)
        m = b['am'].unsqueeze(-1).float()
        zt = s.ln(((ht + a1) * m).sum(1) / m.sum(1))
        zv = s.ln((hv + a2).mean(1))
        out = s.head(torch.cat([zt, zv, b['tab'], b['has']], 1)).squeeze(1)
        return out, Fn.normalize(zt, dim=-1), Fn.normalize(zv, dim=-1)


def smape_log(p, t):                       # exact SMAPE on log scale (de-standardise first)
    return (2 * torch.tanh(((p - t) * sd).abs() / 2)).mean()


def run_fold(f):
    t, v = np.where(fold != f)[0], np.where(fold == f)[0]
    mk = lambda df, tb, yy, sh: torch.utils.data.DataLoader(DS(df, tb, yy), a['bs'], shuffle=sh,
                                                           collate_fn=collate, num_workers=a['workers'])
    dl = mk(tr.iloc[t], Ttr[t], y[t], True)
    net = Net().to(dev)
    enc = list(net.te.parameters()) + list(net.ve.parameters())
    ids = {id(p) for p in enc}
    opt = torch.optim.AdamW([{'params': enc, 'lr': a['lr']},
                             {'params': [p for p in net.parameters() if id(p) not in ids], 'lr': a['head_lr']}],
                            weight_decay=a['wd'])
    steps = a['ep'] * len(dl)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / max(1, a['warm'] * steps)) *
                                            0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
    sc = torch.amp.GradScaler(enabled=dev == 'cuda')
    for ep in range(a['ep']):
        net.train()
        for b in dl:
            b = {k: x.to(dev) for k, x in b.items()}
            with torch.autocast(dev, enabled=dev == 'cuda'):
                p, zt, zv = net(b)
                base = Fn.huber_loss(p, b['y'], delta=1.0) if ep < a['smape_from'] else \
                    0.5 * Fn.l1_loss(p, b['y']) + smape_log(p, b['y'])
                lg = net.tau.exp() * zt @ zv.T
                lab = torch.arange(len(lg), device=dev)
                aux = (Fn.cross_entropy(lg, lab) + Fn.cross_entropy(lg.T, lab)) / 2
                loss = base + a['aux'] * aux
            opt.zero_grad(); sc.scale(loss).backward(); sc.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); sc.step(opt); sc.update(); sch.step()
        print(f'fold {f} epoch {ep} loss {loss.item():.4f}')

    def pred(df, tb):
        net.eval(); out = []
        with torch.no_grad():
            for b in mk(df, tb, None, False):
                b = {k: x.to(dev) for k, x in b.items()}
                with torch.autocast(dev, enabled=dev == 'cuda'):
                    out.append(net(b)[0].float().cpu().numpy())
        return np.concatenate(out) * sd + mu
    return v, pred(tr.iloc[v], Ttr[v]), pred(te, Tte)


oof = np.full(len(tr), np.nan); tst = []
fs = range(fold.max() + 1) if a['fold'] < 0 else [a['fold']]
for f in fs:
    v, pv, pt = run_fold(f)
    oof[v] = pv; tst.append(pt)
    o = np.full(len(tr), np.nan); o[v] = np.exp(pv)            # full length, NaN outside this fold
    np.save(f"oof_{a['tag']}_f{f}.npy", o); np.save(f"tst_{a['tag']}_f{f}.npy", np.exp(pt))
if a['fold'] < 0:
    from metrics import smape
    print('OOF smape', smape(np.exp(y), np.exp(oof)))
    np.save(f"oof_{a['tag']}.npy", np.exp(oof)); np.save(f"tst_{a['tag']}.npy", np.exp(np.mean(tst, 0)))
