#!/bin/bash
#SBATCH -J er_par
#SBATCH -p mtech
#SBATCH -c 24
#SBATCH --mem=90G
#SBATCH -t 04:00:00
#SBATCH -o /csehome/m25csa019/amazon_ml/er/logs/%x_%j.out
# CMDS_A and CMDS_B: two ';'-separated python command lists run in parallel (half the CPUs each, logs <job>_a/_b.log),
# then CMDS (sequential, all CPUs). One QOS slot for two stage-2 trainings.
source /csehome/m25csa019/amazon_ml/er/jobs/env.sh
L=/csehome/m25csa019/amazon_ml/er/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
run() { IFS=';' read -ra C <<< "$1"; for c in "${C[@]}"; do echo "=== $c"; eval "$PY $c" || return 1; done; }
N=${SLURM_CPUS_PER_TASK:-24}
(export SLURM_CPUS_PER_TASK=$((N / 2)) POLARS_MAX_THREADS=$((N / 2)); run "$CMDS_A") > ${L}_a.log 2>&1 & a=$!
(export SLURM_CPUS_PER_TASK=$((N / 2)) POLARS_MAX_THREADS=$((N / 2)); run "$CMDS_B") > ${L}_b.log 2>&1 & b=$!
wait $a || { echo "A failed"; exit 1; }
wait $b || { echo "B failed"; exit 1; }
run "$CMDS"
