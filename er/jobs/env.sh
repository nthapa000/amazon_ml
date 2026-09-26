# Paths for the pipeline; every value can be overridden from the environment (defaults = the author's cluster).
export ER_DATA=${ER_DATA:-/csehome/m25csa019/amazon_ml/student_resource/dataset}   # dir with train/ and test/
export ER_WORK=${ER_WORK:-/scratch/m25csa019/er_work}                              # intermediates (~200 GB)
export HF_HOME=${HF_HOME:-/scratch/m25csa019/hf}                                   # pretrained weights cache
export PY=${PY:-/scratch/m25csa019/envs/er/bin/python}
export POLARS_MAX_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd "$(dirname "${BASH_SOURCE[0]}")/.."
