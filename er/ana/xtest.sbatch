#!/bin/bash
#SBATCH -J xtest
#SBATCH -p mtech
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 00:40:00
#SBATCH -o /csehome/m25csa019/amazon_ml/er/logs/%x_%j.out
# Local end-to-end test of an exported stage-2 package (tar of oof_/pte_/dec_<TAG> + manifest), checked the way imported
# teammate files are: tar md5 -> extract to a fresh dir -> kcheck gate + exact equality with the source files -> tsim
# score with the SHIPPED dec (no re-tuning; per fold via cmp_folds) -> rebuild the upload from the shipped pte + dec + fr
# and compare it with the released matching_results.tsv -> official validator. Reads SRC/REL only; writes to T.
# sbatch ana/xtest.sbatch   (other package: sbatch --export=ALL,TAR=..,MD5=..,TAG=..,REL=..,FR=.. ana/xtest.sbatch)
source /csehome/m25csa019/amazon_ml/er/jobs/env.sh
export SRC=${SRC:-/scratch/m25csa019/er_final}
export TAR=${TAR:-$SRC/export_for_kushal.tar}
export MD5=${MD5:-dd3019ea1d4c189fee0a432283d02975}
export TAG=${TAG:-fin5}
export REL=${REL:-/scratch/m25csa019/versions/fin5_frm2}
export FR=${FR:--2}
export T=${T:-/scratch/m25csa019/xtest_$TAG}
fail=0

echo "== 1. tar integrity (expect md5 $MD5)"
got=$(md5sum "$TAR" | cut -d' ' -f1); echo "md5 $got"
[ "$got" = "$MD5" ] && echo "md5 OK" || { echo "md5 FAIL"; fail=1; }
mkdir -p $T/w $T/out && tar -xf "$TAR" -C $T || { echo "extract FAIL"; exit 1; }
D=$T/$(tar -tf "$TAR" | head -1 | cut -d/ -f1); ls -la $D
# test work dir: shipped files as tag $TAG, the original source files as tag ref, shared read-only inputs from SRC
for k in oof pte; do ln -sfn $D/${k}_$TAG.parquet $T/w/${k}_$TAG.parquet; ln -sfn $SRC/${k}_$TAG.parquet $T/w/${k}_ref.parquet; done
ln -sfn $D/dec_$TAG.json $T/w/dec_$TAG.json; ln -sfn $SRC/dec_$TAG.json $T/w/dec_ref.json
for f in n_test_s1 n_train_s1 n_tsim_s1 gt_pairs_tsim; do ln -sfn $SRC/$f.parquet $T/w/$f.parquet; done
export ER_WORK=$T/w

echo "== 2. gate: ana/kcheck.py (rows, key overlap, p range, corr) + exact equality with the source files"
$PY ana/kcheck.py tag=$TAG ref=ref || fail=1
$PY - <<'PY' || fail=1
import os, polars as pl
W, t = os.environ['ER_WORK'], os.environ['TAG']
ok = True
for k in ('oof', 'pte'):
    a = pl.read_parquet(f'{W}/{k}_{t}.parquet'); b = pl.read_parquet(f'{W}/{k}_ref.parquet', columns=['id1', 'id2', 'p'])
    same = a.columns == ['id1', 'id2', 'p'] and a.height == b.height and bool((a['id1'] == b['id1']).all()) \
        and bool((a['id2'] == b['id2']).all())
    dp = float((a['p'] - b['p']).abs().max()) if same else float('nan')
    print(k, 'rows', a.height, 'dtypes', a.dtypes, 'same ids + row order', same, 'max|dp|', dp, flush=True)
    ok &= same and dp == 0.0
raise SystemExit(0 if ok else 1)
PY

echo "== 3. tsim score of the shipped oof with the shipped dec (decide.select + decide.score, no re-tuning)"
$PY - <<'PY' || fail=1
import os, sys, json, polars as pl
sys.path.insert(0, 'src')
from block import enc, gt_path
from decide import select, score
W, t = os.environ['ER_WORK'], os.environ['TAG']
cfg = json.load(open(f'{W}/dec_{t}.json'))
s1 = pl.read_parquet(f'{W}/n_tsim_s1.parquet', columns=['id', 'country']).select(enc('id').alias('s1'), 'country')
gt = pl.read_parquet(gt_path('tsim')).select(enc('s1').alias('s1'), enc('id2').alias('id2'))
r = score(select(pl.read_parquet(f'{W}/oof_{t}.parquet'), cfg), s1, gt)
print('decision', {k: v for k, v in cfg.items() if k not in ('oof', 'split')}, 'split', cfg['split'])
for k in r:
    print(f'{k:10s} shipped {r[k]:.7f}   stored in dec {cfg["oof"][k]:.7f}   diff {r[k] - cfg["oof"][k]:+.1e}', flush=True)
raise SystemExit(0 if all(abs(r[k] - cfg['oof'][k]) < 1e-9 for k in r) else 1)
PY
$PY ana/cmp_folds.py a=ref b=$TAG || fail=1

echo "== 4. rebuild the upload from the shipped pte + dec + fr=$FR, compare with $REL"
$PY src/decide.py apply tag=$TAG out=$T/out fr=$FR nocand=1 || fail=1
if cmp -s $T/out/matching_results.tsv $REL/matching_results.tsv; then
  echo "matching_results.tsv BYTE-IDENTICAL to $REL/matching_results.tsv"
else
  echo "matching_results.tsv DIFFERS from $REL"; fail=1
fi
$PY ana/diff_country.py a=$T/out b=$REL
$PY - <<'PY' || fail=1
import os, json
a = json.load(open(f"{os.environ['T']}/out/test_stats.json"))
b = json.load(open(f"{os.environ['REL']}/meta.json"))['test']
print('rebuilt ', a); print('released', b)
raise SystemExit(0 if (a['links'], a['rows']) == (b['links'], b['rows']) else 1)
PY
$PY /csehome/m25csa019/amazon_ml/student_resource/utils/validate_submission.py --matching $T/out/matching_results.tsv \
  --candidate $REL/candidate_pairs.tsv --test-dir $ER_DATA/test || fail=1

echo "== RESULT: $([ $fail = 0 ] && echo 'ALL CHECKS PASS' || echo 'SOME CHECK FAILED')"
exit $fail
