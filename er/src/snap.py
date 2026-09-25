"""Store a version: submission + OOF metrics + decision config + code snapshot, and append to VERSIONS.md.

python snap.py tag=t2 desc="stage-2 ctx features" [lb=0.981]
-> /scratch/m25csa019/versions/<tag>/{matching_results.tsv, meta.json, code.tar.gz}
"""
import os
import sys
import json
import shutil
import tarfile
import datetime

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')
V = os.environ.get('ER_VERS', '/scratch/m25csa019/versions')
OUT = os.environ.get('ER_OUT', '/scratch/m25csa019/er_out')
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

kv = dict(a.split('=', 1) for a in sys.argv[1:])
tag = kv['tag']
d = f'{V}/{tag}'
os.makedirs(d, exist_ok=True)
src = f'{OUT}/{tag}/matching_results.tsv'
if os.path.exists(src):
    shutil.copy2(src, f'{d}/matching_results.tsv')
    os.makedirs(f'{REPO}/submissions/{tag}', exist_ok=True)       # repo copy for download / upload
    shutil.copy2(src, f'{REPO}/submissions/{tag}/matching_results.tsv')
dec = json.load(open(f'{W}/dec_{tag}.json')) if os.path.exists(f'{W}/dec_{tag}.json') else {}
ts = json.load(open(f'{OUT}/{tag}/test_stats.json')) if os.path.exists(f'{OUT}/{tag}/test_stats.json') else {}
meta = dict(tag=tag, desc=kv.get('desc', ''), date=datetime.datetime.now().isoformat(timespec='minutes'),
            decision={k: v for k, v in dec.items() if k not in ('oof', 'split')}, oof=dec.get('oof', {}),
            split=dec.get('split', ''), test=ts, lb=kv.get('lb', ''))
json.dump(meta, open(f'{d}/meta.json', 'w'), indent=1)
with tarfile.open(f'{d}/code.tar.gz', 'w:gz') as tf:
    for x in ('src', 'jobs', 'run_all.sh', 'requirements.txt'):
        tf.add(f'{REPO}/{x}', arcname=x)
o = meta['oof']
row = (f"| {tag} | {meta['date']} | {meta['desc']} | {meta['split']} | {o.get('ALL', float('nan')):.5f} | "
       f"{o.get('US', float('nan')):.5f} | {o.get('India', float('nan')):.5f} | {o.get('F0.4', float('nan')):.5f} | "
       f"{ts.get('links', '')} | {meta['lb']} | {d} |\n")
vp = f'{REPO}/VERSIONS.md'
if not os.path.exists(vp):
    open(vp, 'w').write('# Versions\n\nOOF = macro F0.5 (split column says which: train = all train S1, tsim = test-like '
                        'split with 19% S1 dropped; tsim is harder, so compare within the same split).\n\n'
                        '| tag | date | change | split | OOF F0.5 | US | India | F0.4 | test links | LB | stored at |\n'
                        '|---|---|---|---|---|---|---|---|---|---|---|\n')
open(vp, 'a').write(row)
print(row)
