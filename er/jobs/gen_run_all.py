#!/usr/bin/env python3
"""Regenerate run_all.sh from jobs/final.dag (same commands, same order; personal paths -> variables).
usage: python3 jobs/gen_run_all.py"""
import re

DESC = {'base': 'normalize, test-like split, blocking (top-100 caps), pair features',
        'a1': 'stage-1 LightGBM on the K100 candidates',
        'cev': 'cross-encoder 1 (English MiniLM, normalized text) on the a1 uncertain band  [GPU]',
        'kx': 'sibling expansion around confident a1 matches -> kx candidate set + features',
        'ann': 'ANN pass (frozen multilingual-e5-small), validation split  [GPU]',
        'annte': 'ANN pass, test split  [GPU]',
        'b1': 'stage-1 LightGBM on the kx candidates',
        'ka': 'merge ANN candidates (ka set), ANN stage-1 model, context + sibling-support features',
        'cem': 'cross-encoder 2 (multilingual MiniLM, raw text) on the b1ka uncertain band  [GPU]',
        'ceb': 'cross-encoder 3 (bge-reranker-base, raw text) on the b1ka uncertain band  [GPU]',
        'fin': 'cascade stage-2 LightGBM, final candidate set = stage-1 filter p1 >= 0.005, decision rule tuned on OOF, '
               'write outputs, validate'}
SUB = {'/scratch/m25csa019/er_out/finc': '$OUT', '/scratch/m25csa019/er_out/fin': '$OUT',
       '/csehome/m25csa019/amazon_ml/student_resource/utils/validate_submission.py': '$VALIDATOR',
       '/csehome/m25csa019/amazon_ml/student_resource/dataset/test': '$ER_DATA/test'}

steps = []
for ln in open('jobs/final.dag'):
    if ln.strip() and not ln.startswith('#'):
        name, deps, cmd = [x.strip() for x in ln.split('|', 2)]
        steps.append((name, deps, re.search(r'(CMDS|ARGS)="([^"]*)"', cmd).group(2).split(';')))
out = ['#!/bin/bash',
       '# End-to-end pipeline: raw TSVs -> $OUT/matching_results.tsv + $OUT/candidate_pairs.tsv.',
       '# Generated from jobs/final.dag by jobs/gen_run_all.py (same commands, same order). Steps marked [GPU] need one',
       '# CUDA GPU (sm_80/sm_86: A30, A6000, A100). CPU steps: 32 CPUs, ~230 GB RAM. On SLURM: python3 jobs/dag.py jobs/final.dag',
       'set -e',
       'source "$(dirname "$0")/jobs/env.sh"          # ER_DATA, ER_WORK, HF_HOME, PY',
       'OUT=${OUT:-$ER_WORK/out}',
       'VALIDATOR=${VALIDATOR:-$ER_DATA/../utils/validate_submission.py}',
       'mkdir -p "$ER_WORK" "$OUT"', '']
for i, (name, deps, cmds) in enumerate(steps, 1):
    out.append(f'# ---- {i}/{len(steps)} {name}: {DESC[name]}' + (f'   (after: {deps})' if deps else ''))
    for c in cmds:
        for a, b in SUB.items():
            c = c.replace(a, b)
        out.append(f'$PY {c}')
    out.append('')
open('run_all.sh', 'w').write('\n'.join(out))
print(len(steps), 'steps', sum(len(s[2]) for s in steps), 'commands')
