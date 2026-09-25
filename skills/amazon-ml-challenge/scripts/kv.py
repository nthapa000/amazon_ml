"""Tiny arg parser: `python x.py a=1 b=foo c=x,y`. Keeps commands short and flag-free."""
import sys


def args(defaults):
    a = dict(defaults)
    for tok in sys.argv[1:]:
        if '=' not in tok:
            sys.exit(f'bad arg {tok!r}; use key=value. keys: {sorted(defaults)}')
        k, v = tok.split('=', 1)
        if k not in a:
            sys.exit(f'unknown key {k!r}; keys: {sorted(defaults)}')
        d = defaults[k]
        if isinstance(d, bool):
            v = v.lower() in ('1', 'true', 'yes', 'y')
        elif isinstance(d, int):
            v = int(v)
        elif isinstance(d, float):
            v = float(v)
        a[k] = v
    return a


def cols(s):
    return [c for c in s.split(',') if c] if s else []


def read(path):
    import pandas as pd
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def is_text(s):
    import pandas as pd
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
