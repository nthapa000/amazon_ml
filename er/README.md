# Business Entity Resolution — reproducible pipeline

Links every Source-1 business to its Source-2 / Source-3 records using only name, address and country.
Stages: normalization → multi-key blocking → pair features → LightGBM (stage 1) → context features →
LightGBM (stage 2, + cross-encoder score) → one-to-one assignment + F0.5-tuned threshold.

## Environment
```
conda create -p <env> python=3.11
<env>/bin/pip install -r requirements.txt     # torch from the cu121 index
```
Pretrained weights (Apache-2.0): `cross-encoder/ms-marco-MiniLM-L6-v2` (22M params), cached under `$HF_HOME`.
No external data, APIs or lookups are used; all dictionaries are mined from the provided training pairs.

## Paths
Edit `jobs/env.sh`: `ER_DATA` (dataset dir with train/ and test/), `ER_WORK` (scratch for intermediates, ~150 GB),
`HF_HOME`, `PY`.

## Run end-to-end
`bash run_all.sh` (inside an allocation with 32 CPUs, ~230 GB RAM, 1 GPU), or submit the individual
`jobs/*.sbatch` steps. Outputs: `$OUT/matching_results.tsv` and `$OUT/candidate_pairs.tsv`, validated with
`student_resource/utils/validate_submission.py`.

## Source map (`src/`)
| file | role |
|---|---|
| `metric.py` | macro F-beta exactly as the README (unit test: README example = 0.714) |
| `norm.py` | name / address normalization (legal forms, street types, states, transliteration, parsing) |
| `dicts.py` | learns native-script → Latin token dictionary from matched train pairs |
| `prep.py` | TSV → parquet, parallel normalization |
| `mksim.py` | test-like training split: drops 19% of train S1 so their S2/S3 records become orphans |
| `block.py` | compound-key blocking (name×address, address pairs, full name, codes, substrings), IDF key score, top-K caps, recall report |
| `feats.py` | rapidfuzz pair features (name, address, legal form, house numbers, domain, alt names) |
| `feats2.py`, `xf.py` | hard-negative and split-statistics features (initials mismatch, house-number truncation, name commonness, invented-name signal) |
| `train.py` | LightGBM, 5-fold GroupKFold by S1 id, OOF + fold-averaged test predictions |
| `ctx.py` | stage-2 competition features from stage-1 OOF scores |
| `ce.py` | cross-encoder re-scoring of the uncertain band (2-fold OOF) |
| `decide.py` | one-to-one assignment (margin), threshold / expected-F selection tuned on OOF, writes outputs |
