# Experiment log

OOF = macro F0.5 over all 2,206,821 train S1 entities (5-fold GroupKFold by S1 id). Ceiling = F0.5 with a perfect matcher on the candidate set.

| exp | change | pair recall | ceiling F0.5 | cand/S1 | OOF F0.5 | US | India | LB | kept |
|---|---|---|---|---|---|---|---|---|---|
| b1 | blocking v1: single tokens/pairs/tok+state/concat/hn+word, caps 30/300 | 0.921 | 0.955 | 147 | | | | | no |
| b2 | blocking v2: compound keys name x addr-word, addr-word pairs, full name; IDF key score; top-K (K1=40,K2=6) | 0.9525 | 0.9792 | 42 | | | | | no |
| b3 | + full-name x any addr word, raw address codes, digit-only hn, 6-char substrings for concatenated names | 0.9728 | 0.9911 | 42 | | | | | yes |
| s1 | stage-1 LGBM (65 feats: rapidfuzz name/addr, key evidence r1/r2/ks), frac=0.2, one-to-one + thr 0.7 | 0.9728 | 0.9911 | 42 | 0.97803 | 0.97781 | 0.97835 | | yes (sub1) |

| s1x | s1 + xf (name commonness, invented-name vocab signal) | 0.9728 | 0.9911 | 42 | 0.98001 | 0.97980 | 0.98032 | | yes |
| t1 | switch to tsim split (19% S1 dropped -> test-like orphan rate); stage-1 + xf | 0.9743* | 0.9916* | 46 | 0.97961 (tsim) | 0.97943 | 0.97987 | | yes |
| t2 | + stage-2 context/competition feats from t1 OOF (c_marg2 top) | | | | 0.98092 (tsim) | 0.98064 | 0.98133 | | yes (sub2) |
| u1 | stage-1 + xf + feats2 (initials mismatch, house-number distance/truncation) | | | | 0.98117 (tsim) | 0.98148 | 0.98072 | | yes |
| ce | cross-encoder MiniLM-L6 on band 0.02-0.98 (1.05M train pairs), 2-fold OOF: band AUC ce 0.908 vs p 0.911 | | | | | | | | feature |
| u2 | stage-2 on u1: xf + feats2 + ctx(u1) + ce | | | | 0.98422 (tsim) | 0.98398 | 0.98459 | | yes (sub3) |
| ce12 | cross-encoder MiniLM-L12 on band 0.003-0.997 (2.1M pairs): band AUC ce12 0.972 vs p 0.971 | | | | | | | | feature |
| u3 | u2 + ce12 as stage-2 feature (ce12 gain 0.019 vs ce 0.002) | | | | 0.98485 (tsim) | 0.98447 | 0.98542 | | yes (sub4) |
| ce12e2 | L12 cross-encoder 2 epochs: band AUC 0.977 (vs 0.972 1 epoch) | | | | | | | | feature |
| u4 | u3 with ce12e2 | | | | 0.98517 (tsim) | 0.98475 | 0.98580 | | yes (sub5) |
| v1 | stage-1 on K100 candidates (160M pairs, frac 0.15) | 0.9777 | 0.9928 | 90 | 0.98124 (tsim) | 0.98162 | 0.98068 | | stage-1 only |
| exp | sibling expansion (S1 enriched with confident matches, re-block unassigned records): recovers 19.4k/137.7k missed, +12M pairs | 0.9809 | 0.9936 | | | | | | building kx |
| u5 | stage-3: ctx recomputed from u4 scores | | | | 0.98517 (tsim) | 0.98476 | 0.98579 | | no (= u4) |
| dec-grid | u4 decision re-tuned over margin {0,.1,.2,.3} x ef(m,a) x thr: best 0.98519 (margin .3) vs 0.98517 -> plateau, kept margin .2. No one-to-one: 0.98368 (-0.0015) | | | | | | | | no |
| w1 | stage-1 on kx (K100 + sibling expansion), frac .15 | 0.9809 | 0.9936 | 97 | 0.98203 (tsim) | 0.98253 | 0.98129 | | stage-1 only |
| w2 | stage-2 on kx: ctx(w1) + ce(t1 bands) + cev(K100 band) | | | | 0.98634 (tsim) | 0.98599 | 0.98686 | | yes (sub6) |

(*) tsim blocking numbers at K1=40 differ slightly because the S1 set differs. tsim OOF is not directly comparable with train OOF (harder).

Error analysis u3 (tsim, actual decision rule): precision 0.9977, recall 0.9595; singletons correct-empty 98,600/99,652 (98.9%).
Loss 27,076 = missed links 17,540 (65%) + matched entities predicted empty 5,466 (20%) + wrong merges 2,974 (11%) + singleton FP 1,052 (4%).
Missed links 250,643: blocked 158,963 (63%), below threshold 91,680. -> remaining gain is mostly recall (blocking), not precision.

Playbook audit (4H / S5), 2026-09-25:
- Adversarial validation (pair features, tsim vs test): US AUC 0.642, India 0.588 -> mild drift, driven by blocking stats (ks, nk, r1/r2), not similarity features. tsim is a fair proxy.
- France vs US+India: AUC 0.999, 79% from st_eq (state parser knows only US/IN states -> always 'unknown' for FR).
  Simulation (w1 stage-1 fold models, 25% tsim S1, st_eq forced -1): F0.5 0.98122 -> 0.98068 (-0.0005) -> expected LB impact ~0.0001 (France = 15%). Not fixed (low value, risk).
- Transliteration dict coverage of native tokens: train 0.9776 vs test 0.9639 -> small in-sample optimism for India.
- Calibration (w2 OOF): well calibrated at the extremes; slight over-confidence in 0.4-0.8 (e.g. mean p 0.65 -> rate 0.56), absorbed by the tuned decision rule.

Phase 2 (after w2):
- w2a: w2 with frac .5, lr .05 -> OOF 0.98670 (US 0.98643, India 0.98711), +0.00036 vs w2 -> released (sub7).
- Cascade parity: decide on w2 OOF restricted to w1 p>=1e-3 (8.0M of 172.4M pairs, 778 of 6.07M positives lost): 0.986338 vs 0.986337 on all pairs -> cascade is lossless.
- Error analysis w2 (tsim): loss 24,421 = only_fn 15,902 (65%) + matched-predicted-empty 5,155 (21%) + wrong merges 2,441 (10%) + singleton FP 895 (4%).
  precision 0.9980, recall 0.9634; singletons correct-empty 98,757/99,652. Missed links: blocked 118,210 (52%), below threshold 108,366 (48%).
- w3a: w2a + cew (L12 cross-encoder retrained 2 ep on the w1/kx band, AUC 0.9737) -> OOF 0.98676 (US 0.98648, India 0.98718), +0.00006 vs w2a
  -> below the acceptance threshold (>= 0.0001 and > 2x seed noise); not released. cew gain share 0.0071 vs cev 0.0091 (largely redundant).
- c1a: CASCADE stage 2 (only pairs with w1 p>=1e-3: 8.0M tsim / 8.7M test), 100% of S1, lr .05, seed 1 -> OOF 0.98685 (US 0.98662, India 0.98719), +0.00015 vs w2a -> released.
- c1b: same, seed 2 -> 0.98684. Seed noise |c1a-c1b| = 0.000004 -> acceptance threshold stays 0.0001.
- cem: multilingual raw-text cross-encoder (mmarco-mMiniLMv2-L12, Apache-2.0), 2 ep on w1 band: band AUC 0.9782 vs stage-1 0.9716 (best CE so far).
- sup: sibling-support features: 8.0M tsim / 8.7M test pairs, 25.2M / 27.4M sibling comparisons, mean 3.15 siblings.
- exp2: expansion round 2 (from w2a OOF, base kx): recovers 689 / 118,210 missed (0.6%), +2.5M pairs -> NO-GO. K200 ceiling gain (+0.0004) also below the +0.0005 bar -> step 4 dropped.
- c2s: c1a + sibling-support features (sup) -> 0.98714 (US 0.98698, India 0.98740), +0.00029 vs c1a.
- c2m: c1a + multilingual raw-text CE (cem) -> 0.98757 (US 0.98741, India 0.98781), +0.00072 vs c1a.
- c2: c1a + sup + cem -> 0.98776 (US 0.98766, India 0.98792, F0.4 0.99000), +0.00091 vs c1a -> released. Top gain: c_marg2 .47, c_p .27, c_gap1 .13, cem .045.
- Error review c2 (tsim): loss 21,869; precision 0.99846, recall 0.96696; singletons correct 99,092/99,652 (99.4%); missed: blocked 118,988, below-threshold 85,406.
  Patterns: (1) low-p misses dominated by NAME TRUNCATION (S2/S3 keeps first words: 'Green Creative Dynamics' -> 'Green Creative'), usually no address;
  (2) near misses: exact name + empty address where same-name S1/orphans exist (largely inherent); (3) FPs: same-name S1 in other states, invented names at shared addresses (inherent).
  -> c4: prefix-ambiguity features (xf2: p_is, p_rev, p_n1 = #S1 starting with the prefix, p_nc) + source-aware sibling counts (sup2: s_nsame).
- c4: c2 + xf2 prefix-ambiguity features + sup2 (source-aware sibling counts) -> 0.98783 (US 0.98773, India 0.98798), +0.00007 vs c2 -> below bar, not released (prefix features not in top-20 gain; truncated-name misses mostly inherently ambiguous).
- CE band coverage check (c2 errors vs stage-1 band 0.003-0.997): FPs 93% in band (562+63 above 0.997), in-candidate FNs 98.5% in band -> widening the CE band not needed.
- gate (entity-level P(no match), LightGBM 5-fold OOF on c2 scores + entity features): AUC 0.99935, but best g=0.85 gives 0.987777 vs 0.987765 (+0.00001) -> not adopted; empty-list rule already near-optimal.
- Official validator with --check-ids on w2a (matching + 2.1 GB candidate file): PASS.

Notes
- s1 decision: thr grid 0.3..0.7 rising monotonically (0.9742->0.9780); expected-F 0.9780; one-to-one +0.0017 vs none at t=0.5. F0.4=0.9815. Test: 5.74M links, pred-empty 5.9% IN / 5.8% US / 5.4% FR.
- b3 uncapped: recall 0.9793 / ceiling 0.9933 at 128 cand/S1 -> ~0.6 pt recall lost to the key-score cap; a learned pre-ranker should recover part of it.
- Remaining misses: very common names (>60 S1 share the core name) with empty/city-only S2 address; DBA names with partial address.
