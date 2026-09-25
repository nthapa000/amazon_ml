"""Merge per-fold outputs (oof_<tag>_f<k>.npy / tst_<tag>_f<k>.npy) produced on different GPUs.

python merge_folds.py tag=mm1 k=5    -> oof_mm1.npy (full OOF) and tst_mm1.npy (geometric mean of folds)
"""
import numpy as np
from kv import args

a = args({'tag': 'mm', 'k': 5, 'geo': True})
O = np.stack([np.load(f"oof_{a['tag']}_f{f}.npy") for f in range(a['k'])])
S = np.stack([np.load(f"tst_{a['tag']}_f{f}.npy") for f in range(a['k'])])
oof = np.nanmax(np.where(np.isnan(O), -np.inf, O), 0)
assert np.isfinite(oof).all(), 'some rows have no OOF prediction: a fold is missing'
tst = np.exp(np.log(S).mean(0)) if a['geo'] else S.mean(0)
np.save(f"oof_{a['tag']}.npy", oof); np.save(f"tst_{a['tag']}.npy", tst)
print('merged', a['k'], 'folds ->', f"oof_{a['tag']}.npy", f"tst_{a['tag']}.npy")
