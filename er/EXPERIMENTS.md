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
- c2x: XGBoost stage-2 member (depth 10, lr .05, same c2 features) -> 0.98778 (on par with c2); blend step failed (comma in sbatch args) -> rerun in chain.
- c2L: c2 without legacy CE features (ce, ce12e2, cev) -> 0.98763 (-0.00013 vs c2) -> legacy CEs still add value; keep for now.
- ANN recall probe (multilingual-e5-small, MIT; normalized 'core | address'; top-k S1 per S2/S3 record within country, GPU exact search), tsim vs base kx:
  top-1 recovers 43,958 / 118,210 base-missed true links (37.2%) at 1.0 new pair per S1; top-3 51,398 (43.5%) at 9.0/S1;
  top-5 55,011 (46.5%) at 18.3/S1; top-10 60,684 (51.3%) at 43.6/S1  -> GO (bar was 15%). Integrating top-3 ("ka" = kx + ANN) with its own stage-1 (w1ann).
- ceb: BAAI/bge-reranker-base (MIT, 278M, XLM-R base) raw-text CE, 2 ep, w1 band: band AUC 0.9804 (best CE; cem 0.9782). -> c6 = c2 + ceb running.
- ANN chain: step 2 (er_kaprep 41084) died after 2 s with signal exit 0:53 before writing a log (transient node issue); resubmitted 20:23 as 41114, running normally.
- c6: c2 + ceb (bge-reranker-base raw-text CE) -> 0.98790 (US 0.98781, India 0.98804, F0.4 0.99013), +0.00014 vs c2 -> released. ceb gain .036 > cem .014.
- w2 vs c2 on test: 57,590 S1 rows differ (3.3%); France 32,157 (12% of FR rows); c2 makes 2.4% fewer FR links. Manual review: most FR links dropped by c2 are
  same-address 'one generic French word swapped' distractors (Nantes College/Amicale, Lille Ecole/Sportive, Don Ecole/Pharmacie) -> precision gain from raw-text multilingual CE.
- LEADERBOARD: c2 = 0.982 (OOF tsim 0.98776) -> gap ~0.006. US/India test ~ tsim (adversarial AUC .59-.64) -> France (15%, unseen) likely ~0.95.
- France EDA (test inputs): (1) 'N°24' -> 'ndeg24' breaks house numbers (23.7k FR S2/S3 records); (2) S1 uses region (86.6%) but S2/S3 use department
  ~28% (Gironde vs Nouvelle-Aquitaine) -> lower address similarity, no state; (3) hyphenated places (Saint-Nazaire vs SAINT NAZAIRE) in ~47% of FR S2/S3;
  (4) Ets/EI legal+abbrev; (5) French ordinals 1er/2e/3eme. Fix: France-only rules in norm.py (dept->region + state, N° strip, hyphen/apostrophe split,
  ordinals, FR legal forms, name synonyms). Verified: 0 of 300,000 US/India records change; 76.8% of FR S2 records change.
- Probe prepared: submissions/probe_c6_noFR (c6 with FR rows emptied) -> F_FR ~= (LB(c6) - LB(probe)) / 0.15 + 0.058.
- Consolidated end-to-end pipeline (jobs/final.dag, work dir er_final) launched 20:55 with FR fixes: K100 -> stage-1 a1 -> expansion -> kx -> ANN top-3 -> ka;
  stage-1 b1 (kx) + b1ann -> ctx/sup; CEs cev (a1 band, normalized), cem + ceb (union band, raw); cascade stage 2 'fin'. CE weights now saved.
- LEADERBOARD (public): c2 0.982, c6 0.9835, probe_c6_noFR (French rows empty) 0.849.
  Decomposition (France weight 0.1497, French singleton rate ~0.057): France(c6) ~= 0.1345/0.1497 + 0.057 ~= 0.955;
  US+India(c6) ~= (0.9835 - 0.1497*0.955)/0.8503 ~= 0.9885 (OOF 0.9879 -> validation faithful). France(c2) ~= 0.946 -> c6's raw-text CE (ceb) gave France ~+0.009.
  => France is the whole LB gap (~0.005 of LB); language-aware (raw-text multilingual) models move France.
- c5: cascade on ka (kx + ANN top-3; own ANN stage-1 w1ann; union ctx/sup; cemka = raw multilingual CE on union band, AUC 0.9790) -> OOF 0.99077
  (US 0.99036, India 0.99138, F0.4 0.99245), +0.00287 vs c6 -> released (biggest single jump). Cascade positives 6,117,930 (+51,132 vs c2).
- COMPLIANCE HARDENING (France): the hand-written department->region table in norm.py was replaced by a mapping LEARNED from the provided
  data (dicts.learn_geo: S1 addresses end with a region, S2/S3 sometimes with a sub-unit; sub-unit -> region of co-occurring cities).
  Learned: gironde->nouvelle aquitaine, nord->hauts de france, loire atlantique->pays de la loire, pas de calais->hauts de france (99%+ purity).
  Re-normalizing 300k French test records reproduces the running final pipeline except 3 records (Merignac, Charente; too little support to learn).
- REVERTED (user decision, 2026-09-26): French department->region back to the RULE-BASED table (fits 'rule-based or learned from training data';
  the learned version used test inputs). Score effect 0: the final run 'fin' was normalized with the rule-based table; re-normalizing 300,049 French
  records with the reverted code reproduces fin exactly (0 differences). learn_geo removed from dicts.py; geo hooks removed from norm.py / prep.py.
- User decision (France scope): normalization fixes + France-only threshold probes on the leaderboard (decide.py apply fr=<delta>: p' = sigmoid(logit p + delta)
  for French S1 only; US/India rows identical). NO pseudo-label training on test inputs. Verified: fr=-1 on c5 changes 11,275 French rows, 0 US/India rows.
- LEADERBOARD: c5 = 0.987 (best so far). Implied (US+India ~0.9914 from OOF gain): France ~0.962 (c6 ~0.955) -> ANN helped France too.
- Official validator with --check-ids on w2a (matching + 2.1 GB candidate file): PASS.
- Final run (er_final), 2026-09-26: ka hit the /scratch quota at 02:33 (union test); resumed 02:48 via jobs/resume.dag after deleting 24G of
  consumed intermediates. Disk fix, results-identical: annka union (pred=, thr=) and ctx (thr=) now write only rows with stage-1 p >= 0.0005.
  The cascade stage 2 reads only p >= 0.001, and ctx features are still computed over all pairs. Synthetic test: every row stage 2 reads is
  identical (threshold edges, partial sib coverage, ctx), cand_test_ka is unchanged, and union is re-runnable. Saves ~25G (union ~9G/split, ctx ~13G).

- Improvement round 2026-09-26 (user choice: Optuna + 4th cross-encoder). LB fin = 0.986773 vs c5 0.986624 (+0.00015; US/India OOF +0.00012 explains ~+0.0001 -> France ~0.9595 -> ~0.9598: France normalization fixes ~neutral). fin test confidence: France 0.41 uncertain pairs/S1 (0.1<p<0.9) vs US 0.14 / India 0.11 (validation 0.18 / 0.11); top-1 in 0.5-0.9: 0.47% vs 0.14% / 0.08%.
- Optuna (tune.py, 30 trials, fold-0 logloss, 19 min): default 0.029538 -> best 0.029291 (num_leaves 74, min_data 240, feature_fraction 0.51, lambda_l2 27.8, min_gain 0.14; top-5 all 'smaller + more regularized'). 5-fold refit fin2b: OOF 0.990908 vs fin 0.990894 (+0.000014; US +0.000009, India +0.000022), 1400-1750 rounds/fold (42 min vs 19) -> below the 0.0001 bar, NOT adopted.
- France calibration probes from fin (decide.py apply fr=+-1, nocand): fr=-1 changes 9,932 French rows, links -9,689 (-1.1%), French empty 5.48% -> 5.62%; fr=+1 changes 9,750, links +8,697, empty -> 5.33%; 0 US/India rows changed; validator PASS. submissions/probe_fin_frm1, _frp1.

- CANDIDATE-SIZE UPDATE (organizers, 2026-09-26: smaller candidate set per S1 ranks higher; candidate code reviewed). ana/cands.py on tsim (final model restricted to each set): first pass 103.3 cand/S1 test, recall 0.98920; stage-1 p1>=0.001 5.30 (F +0); 0.002 4.89 (-0.000004); 0.005 4.47 (-0.000041); 0.01 4.21 (-0.000095); 0.02 3.99 (-0.00018); top-5 4.12 (-0.0056). User chose p1>=0.005 -> src/cands.py; release finc: 7,739,003 test pairs (4.47/S1), candidate_pairs.tsv 2.3 GB -> 122 MB, OOF 0.990853, links 5,842,048, validator PASS.

- fin2b bias check (ana/cmp_folds.py): folds 1-4 (not seen by Optuna) +0.000020, fold 0 -0.000010, per fold -0.000010 +0.000045 -0.000013 +0.000036 +0.000013 -> gain not from selection; ADOPTED (user preference, positive expected value). Release fin2bc = fin2b + cands p1>=0.005: OOF 0.990870 (US 0.990454, India 0.991493), validator PASS. Current best.

- LB probe_fin_frm1 (fin + France shift fr=-1, only French rows differ) = 0.986928 vs fin 0.986773 -> +0.000155 = France F +0.00104 (~0.9598 -> ~0.9608); ~1/3 of the removed borderline French links were wrong (break-even 22%) -> France over-matches. Public LB top-3 (user, 2026-09-26 ~18:00): 0.990748 / 0.990290 / 0.989934 -> with our US/India (~0.9915) rank 1 implies France ~0.9865 vs ours ~0.961: the gap is France.
- French borderline pairs (inspection of unlabeled test inputs, p 0.35-0.75): (a) same address, one generic name word swapped (ZE Ecole SARL vs ZE Club SARL; Quatre Loisirs vs QUATRE SCI & ASSOCIES), (b) same name, different street or house number +-1 with another legal form (Dunkerque Maternelle 44 Rue de Normandie vs 16 Rue de Roeun; Calais Comite SAS 234 vs SA 235), (c) invented DBA names at the S1 address (Iriirilum, Yumacalo), (d) exact-name records with EMPTY address (Pessac College SARL, Locale Parents SARL): these look like true matches that a uniform shift also suppresses.

- LB probe_fin_frm2 (fin + fr=-2) = 0.986946 vs fr=-1 0.986928 (+0.000018) vs fin 0.986773 (+0.000173). The 11,670 extra French links removed by -2 vs -1 were ~23% wrong (break-even 22%) -> optimum ~fr=-2, further shifts would not pay. France F ~0.9608 -> ~0.9609. fr=-2 makes the French predicted-empty rate 5.78% = US/India. ADOPTED fr=-2 for the final model.

- LLM judge smoke test (Qwen2.5-7B-Instruct, Apache-2.0, zero-shot yes/no logit, 2,000 labelled US/India band pairs of fin2b, A30 ~40 pairs/s): AUC llm 0.531 vs final-model p 0.772; 5-fold stack p+llm 0.770 vs p 0.772 -> no signal; the LLM says No to almost everything (mean -15.4 true vs -16.3 false). The synthetic noise swaps/changes words in TRUE matches (84.6% of borderline word-swap links are correct on US/India), which a semantic judge rejects. Fails the acceptance rule -> LLM lever DROPPED; the narrow France rule is dropped for the same reason.

- Normalization-gap audit (22:30, read-only; final-model test pairs; one-word differences classified): borderline pairs with a one-word name difference: France typo 0.96% / abbreviation 0.24% / glued number 0.02% / DIFFERENT WORD 22.8%; India 2.65/0.49/0.03/9.3%; US 3.58/0.52/0.05/8.9%. Typos/abbreviations are generator noise in every country (fewer in France) and already tolerated; France's excess is generic-word swaps (club->comite/fils/compagnie), also frequent among confident French pairs -> not fixable by normalization. Rule-worthy gaps: FR freres/frs, appartement/apt, glued house-number suffixes (2bis, 26b), avenue/allee typos (~1.2% of French borderline pairs); US ambiguous expansion St->street/saint, CT->court/Connecticut, MT->mount/Montana, NE, WY (~0.4% of US borderline). Expected LB ~+0.00007 vs a 9-12 h full rerun -> NOT rerun (user); documented as limitations / next steps.

- Q&A review (26 Sep 23:10, organizers' answers): self-training / pseudo-labels on test and synthetic pairs from provided records ALLOWED (25 Sep 02:03 ... 26 Sep 19:05); stricter threshold for countries unseen in training ALLOWED (26 Sep 19:05 Q4); private ranking = private score of the BEST PUBLIC submission (25 Sep 18:36, 26 Sep 12:56, 17:03) -> every upload must be package-consistent (its candidate file contains all its matches); candidate size = total pairs (26 Sep 12:56), pruned first-stage-model filter set OK (25 Sep 20:58). decide.py fr= now shifts countries unseen in training (unseen() = test countries not in n_train_s1 -> ['France']), no country named in code.
- Step 0a duplicates (ana/dups.py, fin2): S2/S3 with a content-identical twin 0.86%; twins of selected links left unselected while candidates of the same S1: 17 (of 71,817) -> 'list every identical id' rule changes tsim F0.5 by +0.000000. No change.
- Step 0b/0c L1 self-training set (src/fradapt.py, teacher fin2): tsim precision at the same cutoffs: pseudo-pos (one-to-one margin 0.3, p>=0.97) 0.9996, look-alike pseudo-neg (p<=0.02, max token_set >= 80) 0.9980. France pools: pos 832,717, look-alike neg 920,227 (of 2,107,581 scored pairs); exact S1-S1 negatives 21,427 same address + 106,436 same core name. Set fra_b1ka: 200k pos + 200k neg + 127,863 s1neg (one side noised) + 100k syn (S1 vs noised self) = 627,863.

- cer2 (bge-reranker-v2-m3, 1.6M pairs x 2 epochs per half, Blackwell): band AUC 0.9839 (ceb 0.9804, cer 0.9802). fin3 = fin2 with cer2 instead of cer: OOF 0.991115 (US 0.990703, India 0.991734) vs fin2 0.990957: +0.000159, per fold +0.000146 +0.000192 +0.000179 +0.000101 +0.000176 (folds 1-4 +0.000162) -> ADOPTED. Release fin3c / fin3c_frm2.
- cef (L1) half 0: band AUC 0.9806 (US/India, no regression vs ceb 0.9804) with the French self-training set.
- ceq (00:55, user): Qwen/Qwen3-Reranker-4B (Apache-2.0, 4.02B) as a cross-encoder, LoRA r=16 all-linear + score head, bf16, lr 1e-4, bs 32, max 192 tokens, 1.2M pairs x 2 epochs per half, full band, both halves in one 2-GPU job (jobs/gpu2bw.sbatch, Blackwell-only, jobs/bw_retry.sh). Deadline for everything: 20:00 (user).

Notes
- s1 decision: thr grid 0.3..0.7 rising monotonically (0.9742->0.9780); expected-F 0.9780; one-to-one +0.0017 vs none at t=0.5. F0.4=0.9815. Test: 5.74M links, pred-empty 5.9% IN / 5.8% US / 5.4% FR.
- b3 uncapped: recall 0.9793 / ceiling 0.9933 at 128 cand/S1 -> ~0.6 pt recall lost to the key-score cap; a learned pre-ranker should recover part of it.
- Remaining misses: very common names (>60 S1 share the core name) with empty/city-only S2 address; DBA names with partial address.

- 27 Sep ~03:00 label-free self-check (ana/selfcheck.py, friend's method; beliefs = raw stage-2 p, selection = uploaded one with fr=-2). tsim: F_est 0.99447 vs true 0.99107 (F_est optimistic by ~0.0034: misses outside the candidates are ignored). Test France: fin3c F_est 0.98201, viol (records with sum p > 1) 3.62%, own_rate 0.8233, 3.236 links/S1; fin4c (French CE) F_est 0.98144, viol 3.37% (-7% relative), own_rate 0.8261, 3.243 links/S1. US/India identical. F_est rates the fr=-2 shift as harmful (fin3c France 0.98539 unshifted -> 0.98201) although the LB showed it helps -> self-graded F_est cannot rank France changes; only the viol drop favours fin4c. Inconclusive; LB needed.
- 27 Sep ~04:00 step 3 (job 41443): fin3 seed 2 (fin3s2) 0.991114 vs fin3 0.991115 (seed noise ~0.000001). Blends (logit, weights tuned on OOF, decision re-tuned): b33 fin3+fin3s2 0.991117 (+0.000002), b34 fin3+fin4 (0.6/0.4) 0.991122 (+0.000007), b343 all three 0.991121 (+0.000006). All far below the 0.0001 bar -> no blend adopted; seeds/blends are saturated. Remaining lever: Qwen3-Reranker-4B (ceq).
