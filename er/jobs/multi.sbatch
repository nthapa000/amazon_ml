#!/bin/bash
#SBATCH -J er_multi
#SBATCH -p mtech
#SBATCH -c 32
#SBATCH --mem=200G
#SBATCH -t 08:00:00
#SBATCH -o /csehome/m25csa019/amazon_ml/er/logs/%x_%j.out
# CMDS: ';'-separated python invocations (relative to repo); quotes allowed inside a command
source /csehome/m25csa019/amazon_ml/er/jobs/env.sh
IFS=';' read -ra C <<< "$CMDS"
for c in "${C[@]}"; do echo "=== $c"; eval "$PY $c" || exit 1; done
