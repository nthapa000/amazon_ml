# Versions

OOF = macro F0.5 (split column says which: train = all train S1, tsim = test-like split with 19% S1 dropped; tsim is harder, so compare within the same split).

| tag | date | change | split | OOF F0.5 | US | India | F0.4 | test links | LB | stored at |
|---|---|---|---|---|---|---|---|---|---|---|
| s1 | 2026-09-25T07:17 | stage-1 LGBM 65 feats, blocking b3, thr 0.7 | train | 0.97803 | 0.97781 | 0.97835 | 0.98153 | 5743395 |  | /scratch/m25csa019/versions/s1 |
| t2 | 2026-09-25T07:17 | tsim split + xf + stage-2 context feats, ef margin 0.2 | tsim | 0.98092 | 0.98064 | 0.98133 | 0.98399 | 5760527 |  | /scratch/m25csa019/versions/t2 |
| u2 | 2026-09-25T07:35 | tsim + xf + feats2 hard-negatives + stage-2 ctx(u1) + cross-encoder L6, ef margin 0.2 | tsim | 0.98422 | 0.98398 | 0.98459 | 0.98698 | 5762325 |  | /scratch/m25csa019/versions/u2 |
| u3 | 2026-09-25T07:48 | u2 + cross-encoder L12 wide band (ce12) as stage-2 feature | tsim | 0.98485 | 0.98447 | 0.98542 | 0.98756 | 5766681 |  | /scratch/m25csa019/versions/u3 |
| u4 | 2026-09-25T09:24 | u3 with 2-epoch L12 cross-encoder (band AUC 0.977) | tsim | 0.98517 | 0.98475 | 0.98580 | 0.98785 | 5771704 |  | /scratch/m25csa019/versions/u4 |
| w2 | 2026-09-25T11:49 | K100 caps + sibling expansion (kx) + stage-2 ctx(w1) + cross-encoders (K100 band + t1 bands) | tsim | 0.98634 | 0.98599 | 0.98686 | 0.98880 | 5787290 |  | /scratch/m25csa019/versions/w2 |
| w2a | 2026-09-25T13:10 | w2 trained on 50% of S1 (frac .5, lr .05, seed 1) | tsim | 0.98670 | 0.98643 | 0.98711 | 0.98903 | 5801972 |  | /scratch/m25csa019/versions/w2a |
| c1a | 2026-09-25T14:48 | cascade stage-2 (w1 p>=1e-3, 8.0M pairs), 100% of S1, lr .05, seed 1 | tsim | 0.98685 | 0.98662 | 0.98719 | 0.98921 | 5797106 |  | /scratch/m25csa019/versions/c1a |
| c2 | 2026-09-25T15:27 | cascade + sibling-support features + multilingual raw-text cross-encoder (mmarco-mMiniLMv2) | tsim | 0.98776 | 0.98766 | 0.98792 | 0.99000 | 5781402 |  | /scratch/m25csa019/versions/c2 |
