#!/bin/bash
#SBATCH -J opt_vqc_none_100
#SBATCH -p l40s
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --time=9999:59:59
#SBATCH -o ./logs/260430/%x.o%j
#SBATCH -e ./logs/260430/%x.e%j

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ligand-screen

_D="${SLURM_SUBMIT_DIR:-.}"
if   [ -d "$_D/src" ];      then ROOT="$(cd "$_D"    && pwd)"
elif [ -d "$_D/../src" ];   then ROOT="$(cd "$_D/.." && pwd)"
else                              ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi

export PYTHONPATH=$ROOT
cd "$ROOT"

python scripts/hpo.py \
    --config outputs/runs/vqc_none_100_cv4_opt/config_resolved.yaml \
    --n_trials 200

