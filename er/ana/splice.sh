#!/bin/bash
# Country splice: US/India rows from submission A, unseen-country (France) rows from submission B. Candidates must be identical.
# bash ana/splice.sh <A dir> <B dir> <out dir>
set -e
S1=/csehome/m25csa019/amazon_ml/student_resource/dataset/test/test_source1.tsv
A=$1; B=$2; O=$3
cmp -s $A/candidate_pairs.tsv $B/candidate_pairs.tsv || { echo "candidate files differ"; exit 1; }
mkdir -p $O.tmp
awk -F'\t' 'FILENAME==ARGV[1]{if(FNR>1 && $4=="France") fr[$1]=1; next}
            FILENAME==ARGV[2]{if(FNR>1 && ($1 in fr)) b[$1]=$0; next}
            FNR==1{print; next} {if($1 in fr) print b[$1]; else print}' $S1 $B/matching_results.tsv $A/matching_results.tsv > $O.tmp/matching_results.tsv
cp $A/candidate_pairs.tsv $O.tmp/candidate_pairs.tsv
mv $O.tmp $O
echo "rows: $(wc -l < $O/matching_results.tsv)  same-as-A: $(paste $A/matching_results.tsv $O/matching_results.tsv | awk -F'\t' '$1"\t"$2==$3"\t"$4' | wc -l)  same-as-B: $(paste $B/matching_results.tsv $O/matching_results.tsv | awk -F'\t' '$1"\t"$2==$3"\t"$4' | wc -l)"
