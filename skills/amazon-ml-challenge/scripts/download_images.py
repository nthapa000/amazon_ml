"""Parallel, resumable image downloader. Start this in hour 1; it is often the slowest step.

python download_images.py inp=train.csv col=image_link id=sample_id dir=images workers=64
Files are saved as <dir>/<id>.jpg. Re-running skips files already present. Failures are listed in
<dir>/_failed.csv; the pipeline must still predict those rows (zero image vector + has_image=0).
"""
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from kv import args, read

a = args({'inp': 'train.csv', 'col': 'image_link', 'id': 'sample_id', 'dir': 'images',
          'workers': 64, 'retries': 3, 'timeout': 15})
os.makedirs(a['dir'], exist_ok=True)
df = read(a['inp'])[[a['id'], a['col']]].dropna()
todo = [(i, u) for i, u in df.itertuples(index=False) if not os.path.exists(f"{a['dir']}/{i}.jpg")]
print(f'{len(df)} rows, {len(todo)} to download')


def get(i, u):
    for k in range(a['retries']):
        try:
            req = urllib.request.Request(u, headers={'User-Agent': 'Mozilla/5.0'})
            b = urllib.request.urlopen(req, timeout=a['timeout']).read()
            tmp = f"{a['dir']}/{i}.part"
            open(tmp, 'wb').write(b)
            os.replace(tmp, f"{a['dir']}/{i}.jpg")
            return i, None
        except Exception as e:
            err = str(e)
            time.sleep(1 + k)
    return i, err


bad, t0 = [], time.time()
with ThreadPoolExecutor(a['workers']) as ex:
    fs = [ex.submit(get, i, u) for i, u in todo]
    for n, f in enumerate(as_completed(fs), 1):
        i, e = f.result()
        if e:
            bad.append((i, e))
        if n % 2000 == 0:
            print(f'{n}/{len(todo)}  {n / (time.time() - t0):.0f}/s  failed={len(bad)}')
pd.DataFrame(bad, columns=[a['id'], 'error']).to_csv(f"{a['dir']}/_failed.csv", index=False)
print(f'done in {time.time() - t0:.0f}s, failed {len(bad)} (see _failed.csv)')
