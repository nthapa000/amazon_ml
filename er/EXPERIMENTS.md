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
- 27 Sep ~04:00 step 3 (job 41443): fin3 seed 2 (fin3s2) 0.991114 vs fin3 0.991115 (seed noise ~0.000001). Blends (logit, weights tuned on OOF, decision re-tuned): b33 fin3+fin3s2 0.991117 (+0.000002), b34 fin3+fin4 (0.6/0.4) 0.991122 (+0.000007), b343 all three 0.991121 (+0.000006). All far below the 0.0001 bar. USER DECISION: keep small gains (leaderboard gaps are tiny) -> b34 ADOPTED: release b34c OOF 0.991082 (fin3c 0.991074, fin4c 0.99107), b34c_frm2 PASS; differs from fin3c_frm2 in 2146 rows, from fin4c_frm2 in 3151. Remaining big lever: Qwen3-Reranker-4B (ceq).

- 10:44 Unsupervised last-word probe (ana/noise_words.py, ana/lw_probe.py; no labels used on France, checked against
  labelled US/India first): candidate pairs (stage-1 p>=0.001) whose core names differ ONLY in the last word split into
  three classes by comparing S2/S3-vs-S1 last-word frequency within each country (no labels): 'garbled' (typo-like,
  rapidfuzz ratio 50-99) match rate 0.84-0.90 on tsim; 'noise-word' (last word is one over-represented on S2/S3 vs S1,
  e.g. center/services/partners for US/India, developpement/groupe/participations/holding/international/associes for
  France) match rate 0.78-0.90 on tsim; 'ordinary' (a genuinely different word) match rate only 0.07-0.14 on tsim --
  confirms the CHANGES_kushal_local.md France hypothesis (generic-word swap = same business; distinctive-word swap =
  different business) and gives an unsupervised, France-native word list. France's own class-mean stage-1 p (0.51 /
  0.67 / 0.18) matches the same three-way ordering as US/India, so the pattern transfers.
- 10:44 src/fradapt.py: added generic_suffixes() (unsupervised, from the country's own provided test records only,
  same method as ana/noise_words.py) and two new noise() mutations for the 'syn' synthetic-positive pairs only (s1neg
  unchanged): drop/add/swap a generic last word (rates ~0.08/0.08/0.08, near the measured frequencies) and light
  character transposition on one token (~0.12). Old noise() only touched legal forms/formatting/one dropped char --
  never the dominant truncation/generic-suffix pattern behind low-score true matches. New set fra_b1kb (out= renamed
  from fra_b1ka to keep cef's already-built inputs reproducible): same pos/neg/s1neg pools (unchanged), improved syn.
  Sample-checked: mutations fire correctly ("... SCI" -> "... participations", "Collège" -> "OCLLÈGE").
- 10:44 src/ce.py: two independent additions, CPU-tested against real data (no GPU -- both held by ceq): (1) LoRA
  init= resume fix -- transformers 5.17.0 auto-reloads a saved adapter dir FROZEN (is_trainable=False hard-defaulted)
  and the old `if lora:` line then stacked a second blank adapter on top; fixed via explicit
  PeftModel.from_pretrained(base, dir, is_trainable=True). Verified: old path reloads requires_grad=False, new path
  requires_grad=True with weight values identical to the trained adapter (not reinitialized). (2) unseen=1 option:
  re-score only unseen-country (France) test band pairs -- 908,542 of 3,067,262 (29.6%) -- instead of the full band,
  cutting the continued model's re-score from ~5h to ~1h per half; merge(unseen=1 base=<model>) patches just those
  rows into the base model's oof/pte files (US/India rows byte-identical, OOF file untouched, verified on real
  cef_b1ka files: 0 non-French rows changed, French rows = exact average of the two halves).

- 12:1x Reproducibility fix on the France state-completeness change (flagged by a peer session, amazon-ml-5f):
  norm.py feeds prep -> every n_* parquet -> every downstream feature, so the city->region fallback would silently
  change what run_all.sh reproduces for fin3/fin4/b34c/ceq, all built before it existed. Gated the fallback behind
  ER_CITY_REGION=1 (default off): verified with no env var set, norm_addr() on a bare-city French address returns
  the exact old (empty) state; with the flag set, it returns the learned region. Nothing live has been touched yet
  (n_test_s2/s3.parquet re-normalization, plan step 3, is still pending the user's go-ahead) -- currently this is
  inert code, not applied to any built artifact.

- 27 Sep 18:40 quick fixes from `plan (1).md` (ana/qfix.py; decision layer only, no src/ edits). Q2 on b34c (tsim, base reproduces
  dec_b34c 0.991082 exactly): per-source caps S2<=5 / S3<=6 (train GT maxima; 16 links removed) +0.000001; count prior
  (Poisson-binomial PMF of K reweighted by (train histogram / model-implied mean PMF)^lambda, replaces prod(1-q) and sum q
  in the expected-F rule) lambda 0.5 +0.000001 (3/5 folds), lambda 1 -0.000002 (1/5), with caps +0.000002 / -0.000001.
  All below the +0.00003 gate -> NOT adopted, no version. The model's implied match-count distribution already matches the
  generator on US/India; confirms the decision layer is saturated. Train GT constants: singletons 5.58%, zero-S2 13.04%,
  zero-S3 12.07%, S2/S1 1.674, S3/S1 1.788, P(K=k) 0.056/0.054/0.170/0.241/0.219/0.146/0.075/0.029/0.009/0.002.
- 18:40 jobs/par.sbatch --mem 100G -> 90G before fz_f5 is submitted: once ceqf replaces ceq on cn07 only ~94 GB stays
  unallocated there and no other node has 100 GB free, so f5 (fin5) would have waited indefinitely.
- 18:40 France label-free diag on b34c test (ana/qfix.py diag; seen countries US/India: empty 5.78%, links/S1 3.380, S2 1.638,
  S3 1.741). France vs fr: -3 empty 6.06% links 3.186 | -2 5.89% 3.239 | -1.5 5.82% 3.260 | -1 5.75% 3.279 | 0 5.63% 3.312.
  (1) empty-rate match at fr ~ -1.2 for b34c (fin's LB optimum -2 was where empties matched; newer CE stack is less
  over-confident on France) -> Q1 version at the matched fr for the final model, LB probe only (plateau -1..-3 on fin).
  (2) S2 and S3 deficits are proportional (-4.0% / -4.3% at fr=-2) -> no per-source bias, Q4 per-source fr DROPPED.
  (3) France is under-linked at EVERY fr (3.31 vs 3.38 even at fr=0, with a lower empty rate): partial recall inside
  matched French S1, which no global shift fixes -> model-side (ceqf), not calibration.
- 18:47 ceq COMPLETED: merged band AUC 0.98563 (h0 0.98563, h1 0.98563; stage-1 p 0.9697) -> best CE (cer2 0.9839).
  sup3 (sibling support from fin3 stage-2 OOF/pte, sib_thr 0.9, thr 0 = the exact cascade row set) built: 6.0M siblings,
  26.5M comparisons tsim -> W/sup3_ka_{tsim,test}.parquet (for Q5 = fin5a retrained with sup3 instead of sup).
- 18:5x RESOURCE BLOCK: fz_f5 (41941, 24 CPU) could not fit any node (cn07 left 12 CPU / 27 GB after ceqf 16 CPU / 200 GB
  + other users; scheduler ETA 01:41 Sep 28 = after the deadline). Measured stage-2 peaks: fz_f3 18.2 GB, fz_f4 22.5 GB
  (sacct MaxRSS) -> f5 memory lowered to 60G (scontrol). ceq's 231 GB summed RSS vs its 200 GB limit shows most of it is
  shared memory of the forked loader workers. ceqf 41515 was CANCELLED ~5 min in and resubmitted as 41942 with identical
  ARGS/MERGE via jobs/gpu2bw.sbatch but -c 4 --mem=140G -w cn07 (frees the 24 CPU / 60 GB f5 needs). bw_retry.sh for
  ceqf exits on CANCELLED, so 41942 has no automatic retry if it lands on a non-Blackwell GPU (exit 3). TO VERIFY.
- 20:07 fin5 (stage 2 + ceq): fin5a 0.991275 / fin5b 0.991275 (vs fin3 0.991115: +0.000160); blend fin5 (0.8/0.2) 0.991276;
  fold check vs b34: +0.000154 all folds, every fold positive (+0.000126..+0.000180), US +0.000171, India +0.000129.
  Released fin5c (OOF 0.991231, US 0.990842, India 0.991815; vs b34c 0.991082: +0.000149) and fin5c_frm2 (fr=-2),
  both validator PASS, 7,739,003 candidate pairs (4.47/S1), 5,815,975 links. -> UPLOAD #1 = submissions/fin5c_frm2.
  Expected LB ~0.8503 x (0.991231 + 0.00054) + 0.1497 x F_France = 0.9878 (France 0.965) .. 0.9885 (France 0.970).
- 20:06 fin56.dag w6 submit FAILED (8 h waiter on the default `test` partition, max 1 h) -> f6 skipped by the runner.
  f6 submitted manually with the identical commands: fz_f6 41947, afterok:41942 (ceqf), -c 16 --mem=40G.
- 20:20 released fin5_frm2 = fin5 UN-pruned (candidate_pairs = stage-2 cascade set p1>=0.001, 9,187,520 pairs, 5.30/S1) + fr=-2: OOF 0.991276, 5,816,637 links, validator PASS -> UPLOAD #1 (user). fin5c_frm2 (pruned, 4.47/S1, OOF 0.991231) = upload #2.
- 20:48 Q5 (sibling support recomputed from fin3 stage-2 scores = sup3, fin5a retrained as fin5sa): fin5sa 0.991301 vs fin5a
  0.991275 (+0.000027; folds +0.000036 +0.000032 +0.000044 +0.000021 -0.000000; US +0.000025, India +0.000029); blend
  fin5s (fin5sa 0.8 / fin5b 0.2) 0.991298; pruned fin5sc 0.991250 vs fin5c 0.991231 = +0.000019, 5/5 folds positive
  (+0.000030 +0.000015 +0.000037 +0.000011 +0.000002), US +0.000017, India +0.000021 -> step-3 WINNER (handoff rule: best tsim,
  >= 3/5 folds). Decision tune moved m 0.5 -> 1.0.
  Q3 (fin5 + b34 logit blend, 0.8/0.2): b534c 0.991234 vs fin5c +0.000003, folds -0.000007 +0.000001 +0.000021 -0.000004
  +0.000003 (3/5), US -0.000000 -> NOT adopted.
  Q1 on fin5c: France empty rate matches US/India at fr ~ -0.8 (b34c ~ -1.2; current -2).
- 20:49 q_pc (41950): release fin5sc_frm2, then A1 pcdec tune on fin5sc and fin5c. q_f6s queued by watcher after ceqf:
  fin5sa re-scored with ceqf (fin6sa, predict only) + fin6b -> fin6s -> fin6sc -> fin6sc_frm2 + diff_country + France diag.
- 20:55 LB fin5_frm2 (fin5 unpruned + fr=-2) = 0.987997 = NEW TEAM BEST (prev 0.987862, +0.000135). Implied France ~0.9663 (fin_frm2 0.9615, cx1 0.9639): ceq moved France ~+0.0024 over cx1 and ~+0.005 over fin.
- 20:55 released fin5s_frm2 (unpruned fin5s + fr=-2): tsim 0.991298, PASS, 5,818,944 links; vs fin5_frm2 (LB 0.987997) 4,536 rows differ (FR 2324 / IN 932 / US 1280) -> upload #2 candidate (+A1 if it passes). Step-4 job now also releases unpruned fin6s_frm2 (the LB-proven candidate set) besides fin6sc_frm2.
- 20:57 A1 per-country decision (ana/pcdec.py, 61 configs/country, CV = config chosen on 4 folds): fin5sc +0.000005 (both
  countries picked the SAME config a 0.9 / floor 0.01, i.e. a global tweak; 4/5 folds) but fin5c CV -0.000007 (0/5 folds,
  in-sample +0.000000) -> sign flips between near-identical bases = noise -> A1 NOT adopted, no upload. A5 (runner-up >= 0.9
  for unassigned records) -0.000002 on both -> NOT adopted. Decision layer confirmed saturated.
- 21:05 fin5s vs fin5 (the LB-0.987997 base): +0.000022, 5/5 folds (+0.000031 +0.000021 +0.000041 +0.000018 +0.000002),
  US +0.000021, India +0.000024 -> fin5s_frm2 = UPLOAD #2. France diag on fin5s: France empty rate = US/India (5.78%) at
  fr ~ -0.35 (fin -2, b34c -1.2, fin5c -0.8): newer models are ever less over-confident on France -> one fr probe
  (matched value) on the final fin6s = upload #4. France stays under-linked at every fr (3.31 vs 3.38 links/S1 at fr=0).
- 21:07 step 4 re-plumbed for speed: f6 41947 (fin6a+fin6b + fin6 chain, no longer needed since the upload line is fin5s) cancelled; q_p6b 41953 (fin5b models re-score test with ceqf -> fin6b) and q_p6sa 41954 (fin5sa -> fin6sa) run in parallel afterok ceqf 41942; watcher3 submits q_f6s (blend fin6s -> fin6s_frm2 unpruned + fin6sc_frm2 pruned, diff_country vs fin5s(c)_frm2, France diag on fin6s) afterok both. ETA fin6s_frm2 ~22:25.
- 21:15 L3 (plan (1).md IV.2, per-country stage 2) using the idle CPU slot while ceqf scores: new ana/pcsub.py writes country keep-lists (tsim US 4,928,843 / India 3,265,904 pairs; test US 3,205,383 / India 3,874,556) so src/train.py sub=oof_b1ka_<c>:pte_b1ka_<c>:0.001 trains fin5sa's config per country (same folds). q_l3 41959: pcus || pcin -> blend fin5s+pcus+pcin (fin5sp) -> cmp_folds vs fin5s. France keeps the pooled model. To free a submit slot the pending q_p6b/q_p6sa were cancelled; watcher4 submits one par job q_f6 when ceqf COMPLETES: fin6b || fin6sa re-score, then fin6s_frm2 / fin6sc_frm2 releases, diff_country, France diag, and fin6sp_frm2 (= fin6s + L3) for use only if L3 passes its gate.
- 21:30 L1 gate (ana/xlgate.py on fin5s, tsim): India true pairs latin 2,032,361 recall 0.9720 (blocked 1.53%, missed in cands 1.28%, false-link rate 0.14%) vs native-script 446,706 recall 0.9969 (blocked 0.21%, missed 0.10%, FP 0.03%) -> native-script India is the BEST-served slice; no matched tsim record has unknown native tokens (dictionary learned in-sample). Records with unknown native tokens: tsim 1.55% vs test 2.57% of India S2/S3 (+1.0 pt, all still read raw by cer2/ceq/ceqf) -> L1 FAILS the gate, not built.
- 21:55 N4 text-view TTA on ceqf (ana/ce_tta.py; GPU q_tta 41969): French band test pairs whose S2/S3 raw text has spurious accents / non-ASCII or an all-caps name (432,279 of 908,542 = 47.6%) re-scored by both ceqf halves on a folded view (unidecode + lowercase, both records); score = mean(raw, folded). Merge -> ceqft_b1ka_test; then patch alias ceqft:ceq into fin5sa/fin5b (predict only) -> fin7s_frm2 (France rows only change vs fin6s). cer2f (plan B1) impossible: cer2 was trained with save=0 (no weights).
- 21:58 L3 per-country stage 2: fin5sp (0.4 pooled + 0.3 US model + 0.3 India model) 0.991304 vs fin5s 0.991298 = +0.000006, folds +0.000015 +0.000011 +0.000019 -0.000018 +0.000004, US +0.000013, India -0.000004 -> NOT adopted (country worse, below bar). q_f7 41970 (afterok q_tta): fin7sa/fin7b predict with ceqft -> fin7s_frm2.
- 21:48 fin6s_frm2 released (fin5s + ceqf patched on French test pairs, unpruned, fr=-2): PASS, 5,826,291 links; vs fin5s_frm2
  only France changed (11,336 rows; US/India 0). fin6sc_frm2 (pruned) same, France 11,321. France diag on fin6s: empty rate
  = US/India at fr ~ -1.4 (fin5s -0.35: ceqf makes France more match-happy); under-linking shrinks (-0.119 links/S1 at fr=-2
  vs -0.147 for fin5s).
- 21:56 TEAM MERGE inputs from m25csa011 (export tar md5 52b5fe15faaa08a9cbc55886b51a311e verified; extracted to
  /scratch/m25csa019/friend_import, copied as oof_/pte_/dec_k{bl1,cxQ,blSf}): bl1 tsim 0.991336 (LB 0.987655 -> implied France
  ~0.9637), cxQ tsim 0.991508 (their best; cx + cxs + our ceq + their French bge CE cfr + 1.66M French pseudo-label aux rows),
  blSf 0.991359 (no Qwen). Key overlap with our fin5s 100.000% on tsim and test for all three, tsim labels identical; corr(p)
  cxQ vs fin5s 0.9986. Complementary: theirs stronger on US/India (+0.00021 tsim), ours on France (+0.0026 France F).
  q_mx 41971: blend kcxQ+fin6s -> mx6 (tune, fold checks vs both parents), release mx6_frm2, diff_country, France diag,
  then 3-way kcxQ+fin6s+kblSf -> mx6b.
- 22:04 TEAM MERGE 1: mx6 = 0.7 x friend cxQ + 0.3 x our fin6s (blend.py grid, tuned on tsim OOF): tsim 0.991528 (US 0.991122,
  India 0.992135; m 0.0) vs cxQ +0.000020 (5/5 folds, US +0.000027, India +0.000008) and vs fin6s +0.000230 (5/5).
  Released mx6_frm2: PASS, 5,830,448 links; vs fin6s_frm2 France 13,032 / India 4,179 / US 3,708 rows. France diag: empty
  rate = US/India at fr ~ -1.1. 3-way +blSf (0.7/0.15/0.15) 0.991523 < mx6 -> rejected.
- 22:12 friend 3 (Kushal) export via Google Drive (tar md5 1285a800972b5923f8d64364414a29d6, extracted in isolation to
  /scratch/m25csa019/friend3_import; its extras/code/*.py NOT run or copied): fx = LB 0.987862 (team best before tonight),
  tsim 0.99112, uploaded with fr=-2 AND norm1=1.0 (their decide.py one-to-one normalisation, not in our code); implied
  France ~0.9663. Copied as oof_/pte_/dec_kfx. q_tb3 41973: ana/kcheck.py gate -> ana/wblend.py kcxQ+fin6s+kfx (simplex
  weights on tsim; France = equal weights) -> tb3_frm2 release + diff vs mx6_frm2 + France diag.
- 22:20 q_tb3 (41973): kfx gate PASS (rows exact, 100% keys, labels identical; corr 0.9991 fin6s / 0.9982 kcxQ). wblend best seen weights cxQ 0.7 / fin6s 0.3 / fx 0.0 = 0.991528 (= mx6 exactly on US/India; fx adds nothing on tsim); France = equal thirds. tb3_frm2 validator PASS, 5.8M links; job then failed at snap (';' inside the desc split CMDS) -> redone in 41976. /csehome down to 2.5 GB: 10 superseded submission folders (b34c*, fin2*c*, fin3c*, fin4c*, finc) moved to /scratch/m25csa019/sub_archive with md5 check -> 4.6 GB free.
- 22:52 mx7_frm2 = friend cxQ 0.7 + our fin7s 0.3 (fin7s = fin6s + ceqf text-view TTA; weights re-tuned: same 0.7/0.3): tsim 0.991528 (= mx6), PASS, 5,830,392 links; vs mx6_frm2 only 124 French rows differ (US/India 0) -> practically identical to mx6; TTA's effect is diluted to 0.3 and touched only 969 French rows in fin7s. fin7s_frm2 vs fin6s_frm2: 969 French rows. A separate session/user also released fin7s_frm1.5 (fin7s, fr=-1.5) at 22:50 (jobs q_f7r15 41978 failed on ER_WORK, 41979 OK).
- 23:50 export_for_kushal.tar (fin5 = LB-0.987997 stage 2, raw p, for teammate blend; md5 dd3019ea1d4c189fee0a432283d02975) locally tested with new ana/xtest.sbatch (job 41993, PASS): md5 OK; kcheck rows 8,194,747 / 9,187,520, keys 100%, p bit-identical to er_final (max|dp| 0); shipped oof + shipped dec -> tsim 0.9912756 (US 0.9908797, India 0.9918688), per-fold diff 0; decide.py apply shipped pte + dec + fr=-2 -> matching_results.tsv BYTE-IDENTICAL to versions/fin5_frm2 (5,816,637 links), validator PASS. dec has no fr key (fr=-2 is an apply arg); decide always runs one2one (margin 0.3) before ef.
- 00:27 (28 Sep) architecture.md written for the team best tb5_frm2 (LB 0.98834): shared pipeline -> 3 stage-2 members (fin5s / cxQ / fx) -> logit team blend (US/India 0.7/0.3/0, France thirds) -> decision (fr=-2, one2one 0.3, ef m=0); reproduction commands for fin3 -> sup3 -> fin5sa -> fin5s -> wblend/tb5 -> qfix apply. Documentation.md still has TBDs and a stale final (fin6c, p1>=0.005).
