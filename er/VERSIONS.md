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
| c2 | 2026-09-25T15:27 | cascade + sibling-support features + multilingual raw-text cross-encoder (mmarco-mMiniLMv2) | tsim | 0.98776 | 0.98766 | 0.98792 | 0.99000 | 5781402 | 0.982 | /scratch/m25csa019/versions/c2 |
| c6 | 2026-09-25T20:38 | c2 + bigger multilingual raw-text cross-encoder (bge-reranker-base / MIT / 278M) | tsim | 0.98790 | 0.98781 | 0.98804 | 0.99013 | 5776984 | 0.9835 | /scratch/m25csa019/versions/c6 |
| probe_c6_noFR | 2026-09-25 | diagnostic: c6 with all French rows empty | - | - | - | - | - | - | 0.849 | submissions/probe_c6_noFR |
| c5 | 2026-09-25T23:42 | c2 + ANN top-3 candidates (kx+ANN union ka) + ANN stage-1 + union ctx/sup + cemka | tsim | 0.99077 | 0.99036 | 0.99138 | 0.99245 | 5845092 | 0.986624 | /scratch/m25csa019/versions/c5 |
| fin | 2026-09-26T15:06 | final consolidated pipeline: France-fixed normalization + K100/expansion/ANN + 3 cross-encoders (cev, cem, ceb) + cascade stage 2 | tsim | 0.99089 | 0.99048 | 0.99152 | 0.99257 | 5842950 | 0.986773 | /scratch/m25csa019/versions/fin |
| finc | 2026-09-26T17:34 | fin + final candidate set = stage-1 learned filter p1 >= 0.005 (4.47 cand per S1) | tsim | 0.99085 | 0.99044 | 0.99147 | 0.99254 | 5842048 |  | /scratch/m25csa019/versions/finc |
| fin2bc | 2026-09-26T17:46 | fin2b (Optuna stage-2 params) + final candidate set p1 >= 0.005 (4.47 cand per S1) | tsim | 0.99087 | 0.99045 | 0.99149 | 0.99256 | 5841658 |  | /scratch/m25csa019/versions/fin2bc |
| probe_fin_frm1 | 2026-09-26T16:40 | PROBE: fin + France-only shift fr=-1 (US/India rows identical to fin) | tsim | 0.99089 | | | | 5833261 | 0.986928 | submissions/probe_fin_frm1 |
| probe_fin_frm2 | 2026-09-26T19:30 | PROBE: fin + France-only shift fr=-2 (US/India rows identical to fin) | tsim | 0.99089 | | | | 5821492 | 0.986946 | submissions/probe_fin_frm2 |
| fin2c | 2026-09-27T00:13 | fin2 (Optuna + cer) + final candidate set p1 >= 0.005 (4.47 cand per S1) | tsim | 0.99092 | 0.99051 | 0.99152 | 0.99263 | 5835924 |  | /scratch/m25csa019/versions/fin2c |
| fin3c | 2026-09-27T00:58 | fin3 (Optuna + cer2) + final candidate set p1 >= 0.005 | tsim | 0.99107 | 0.99067 | 0.99168 | 0.99272 | 5837080 |  | /scratch/m25csa019/versions/fin3c |
| fin3_frm2 | 2026-09-27T01:27 | fin3 + fr=-2, full candidate set (103/S1, 176.7M pairs; tsv on /scratch) | tsim | 0.99112 | 0.99070 | 0.99173 | 0.99275 | 5819045 |  | /scratch/m25csa019/versions/fin3_frm2 |
| fin4c | 2026-09-27T02:10 | fin3 with cef (French self-trained ceb) + final candidate set p1 >= 0.005 | tsim | 0.99107 | 0.99066 | 0.99168 | 0.99272 | 5840226 |  | /scratch/m25csa019/versions/fin4c |
| fin4c_frm2 | 2026-09-27T02:14 | fin4c + fr=-2 (unseen-country shift); vs fin3c_frm2: 4020 France / 633 India / 597 US rows differ | tsim | 0.99107 | 0.99066 | 0.99168 | 0.99272 |  |  | /scratch/m25csa019/versions/fin4c_frm2 |
| b34c | 2026-09-27T04:16 | blend fin3 0.6 + fin4 0.4 (logit) + final candidate set p1 >= 0.005 | tsim | 0.99108 | 0.99068 | 0.99169 | 0.99273 | 5838049 |  | /scratch/m25csa019/versions/b34c |
| b34c_frm2 | 2026-09-27T04:17 | b34c + fr=-2 (unseen-country shift) - vs fin3c_frm2 2146 rows differ - vs fin4c_frm2 3151 | tsim | 0.99108 | 0.99068 | 0.99169 | 0.99273 | 5819024 |  | /scratch/m25csa019/versions/b34c_frm2 |
| fin5c | 2026-09-27T20:07 | blend fin3+ceq / fin4+ceq (logit) + final candidate set p1 >= 0.005 | tsim | 0.99123 | 0.99084 | 0.99181 | 0.99289 | 5835664 |  | /scratch/m25csa019/versions/fin5c |
| fin5c_frm2 | 2026-09-27T20:07 | fin5c + fr=-2 (unseen-country shift) | tsim | 0.99123 | 0.99084 | 0.99181 | 0.99289 | 5815975 |  | /scratch/m25csa019/versions/fin5c_frm2 |
| fin5_frm2 | 2026-09-27T20:20 | fin5 UN-pruned (candidate set = stage-2 cascade p1>=0.001; 9.19M pairs 5.30/S1) + fr=-2 | tsim | 0.99128 | 0.99088 | 0.99187 | 0.99292 | 5816637 | 0.987997 | /scratch/m25csa019/versions/fin5_frm2 |
| fin5sc_frm2 | 2026-09-27T20:49 | fin5sc = fin5sa (sibling support from fin3 stage-2 scores) + fin5b blend + p1>=0.005 cands + fr=-2 | tsim | 0.99125 | 0.99086 | 0.99184 | 0.99288 | 5818218 |  | /scratch/m25csa019/versions/fin5sc_frm2 |
| fin5s_frm2 | 2026-09-27T20:53 | fin5s UN-pruned (fin5sa sup3 0.8 + fin5b 0.2; candidate set = stage-2 cascade p1>=0.001 5.30/S1) + fr=-2 | tsim | 0.99130 | 0.99090 | 0.99189 | 0.99292 | 5818944 |  | /scratch/m25csa019/versions/fin5s_frm2 |
| fin6s_frm2 | 2026-09-27T21:48 | fin5s UN-pruned with ceqf (French-adapted ceq) patched into unseen-country test pairs + fr=-2 | tsim | 0.99130 | 0.99090 | 0.99189 | 0.99292 | 5826291 |  | /scratch/m25csa019/versions/fin6s_frm2 |
| fin6sc_frm2 | 2026-09-27T21:51 | fin5sc with ceqf (French-adapted ceq) patched into unseen-country test pairs + fr=-2 | tsim | 0.99125 | 0.99086 | 0.99184 | 0.99288 | 5825572 |  | /scratch/m25csa019/versions/fin6sc_frm2 |
| mx6_frm2 | 2026-09-27T22:04 | team merge: logit blend of friend cxQ + our fin6s (weights tuned on tsim OOF) + fr=-2 | tsim | 0.99153 | 0.99112 | 0.99214 | 0.99313 | 5830448 |  | /scratch/m25csa019/versions/mx6_frm2 |
| tb3_frm2 | 2026-09-27T22:22 | 3-person team blend: friend1 cxQ 0.7 + our fin6s 0.3 + friend3 fx 0 on US/India (tuned) and equal thirds on France + fr=-2 | tsim | 0.99153 | 0.99112 | 0.99214 | 0.99313 | 5829789 |  | /scratch/m25csa019/versions/tb3_frm2 |
| fin7s_frm2 | 2026-09-27T22:41 | fin6s with ceqf text-view TTA (accent/case folded view averaged on noisy French pairs) + fr=-2 | tsim | 0.99130 | 0.99090 | 0.99189 | 0.99292 | 5825965 |  | /scratch/m25csa019/versions/fin7s_frm2 |
| fin7s_frm1.5 | 2026-09-27T22:50 | fin7s (fin6s + ceqf text-view TTA) with unseen-country shift fr=-1.5 | tsim | 0.99130 | 0.99090 | 0.99189 | 0.99292 | 5832341 |  | /scratch/m25csa019/versions/fin7s_frm1.5 |
| mx7_frm2 | 2026-09-27T22:52 | team blend: friend cxQ 0.7 + our fin7s 0.3 (fin6s + ceqf text-view TTA on French) + fr=-2 | tsim | 0.99153 | 0.99112 | 0.99214 | 0.99313 | 5830392 |  | /scratch/m25csa019/versions/mx7_frm2 |
| mx7c_frm2 | 2026-09-27T23:10 | mx7 team blend restricted to final candidate set p1 >= 0.005 (7.74M pairs 4.47 per S1) + fr=-2 | tsim | 0.99148 | 0.99108 | 0.99208 | 0.99310 | 5829709 |  | /scratch/m25csa019/versions/mx7c_frm2 |
| tb3c_frm2 | 2026-09-27T23:13 | tb3 team blend restricted to final candidate set p1 >= 0.005 + fr=-2 | tsim | 0.99148 | 0.99108 | 0.99208 | 0.99310 | 5829114 |  | /scratch/m25csa019/versions/tb3c_frm2 |
| mx7c_frm1.5 | 2026-09-27T23:15 | mx7c (team blend on final candidate set p1 >= 0.005) with unseen-country shift fr=-1.5 | tsim | 0.99148 | 0.99108 | 0.99208 | 0.99310 | 5832192 |  | /scratch/m25csa019/versions/mx7c_frm1.5 |
| mx7c_frm1 | 2026-09-27T23:28 | mx7c (team blend on final candidate set p1 >= 0.005) with unseen-country shift fr=-1 (label-free matched point) | tsim | 0.99148 | 0.99108 | 0.99208 | 0.99310 | 5835126 |  | /scratch/m25csa019/versions/mx7c_frm1 |
| tb3_frm1.5 | 2026-09-27T23:31 | tb3 (cxQ+fin6s+fx team blend) with unseen-country shift fr=-1.5 | tsim | 0.99153 | 0.99112 | 0.99214 | 0.99313 | 5833068 |  | /scratch/m25csa019/versions/tb3_frm1.5 |
| mx7c_frm2q | 2026-09-27T23:32 | mx7c_frm2 + decision tweaks: per-source caps S2<=5 S3<=6 and count prior lambda 0.5 | tsim | 0.99148 | 0.99108 | 0.99207 | 0.99309 | 5829673 |  | /scratch/m25csa019/versions/mx7c_frm2q |
| mx7c_frm2k | 2026-09-27T23:34 | mx7c_frm2 + per-source link caps S2<=5 S3<=6 (tsim +0.000000 overall 5/5 folds non-negative) | tsim | 0.99148 | 0.99108 | 0.99208 | 0.99310 | 5829649 |  | /scratch/m25csa019/versions/mx7c_frm2k |
| fin7s_frm2.5 | 2026-09-27T23:43 | fin7s with unseen-country shift fr=-2.5 | tsim | 0.99130 | 0.99090 | 0.99189 | 0.99292 | 5818606 |  | /scratch/m25csa019/versions/fin7s_frm2.5 |
| tb5_frm2 | 2026-09-27T23:48 | tb3 with France member fin5s (plain ceq) instead of fin6s (ceqf drop-in) + fr=-2 | tsim | 0.99153 | 0.99112 | 0.99214 | 0.99313 | 5828829 |  | /scratch/m25csa019/versions/tb5_frm2 |
| tb6_frm2 | 2026-09-27T23:56 | tb5 with France 0.5 cxQ + 0.25 fin5s + 0.25 fx + fr=-2 | tsim | 0.99153 | 0.99112 | 0.99214 | 0.99313 | 5829645 |  | /scratch/m25csa019/versions/tb6_frm2 |
