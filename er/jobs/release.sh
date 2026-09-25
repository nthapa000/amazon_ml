# release_cmds <tag> <desc_without_semicolons> [cand_test_name]  -> CMDS for multi.sbatch: apply, validate, snapshot
release_cmds() {
  local T=$1 D=$2 C=${3:-cand_test} O=/scratch/m25csa019/er_out/$1
  echo "src/decide.py apply tag=$T out=$O cand=$C;/csehome/m25csa019/amazon_ml/student_resource/utils/validate_submission.py --matching $O/matching_results.tsv --candidate $O/candidate_pairs.tsv --test-dir /csehome/m25csa019/amazon_ml/student_resource/dataset/test;src/snap.py tag=$T desc='$D'"
}
