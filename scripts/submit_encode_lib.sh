#!/bin/bash
#SBATCH -J encode
#SBATCH -p l40s
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --time=9999:59:59
#SBATCH -o ./logs/260504/%x.o%j
#SBATCH -e ./logs/260504/%x.e%j

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ligand-screen

_D="${SLURM_SUBMIT_DIR:-.}"
if   [ -d "$_D/src" ];      then ROOT="$(cd "$_D"    && pwd)"
elif [ -d "$_D/../src" ];   then ROOT="$(cd "$_D/.." && pwd)"
else                              ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi

export PYTHONPATH=$ROOT
cd "$ROOT"

python scripts/encode_screening.py \
       --config outputs/runs/exp_best/config_resolved.yaml \
       --screening_csv data/generated_lib.csv \
       --out_dir screening/generated_lib
