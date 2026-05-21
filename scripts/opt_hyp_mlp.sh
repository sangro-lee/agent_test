#!/bin/bash
#SBATCH -J relu_50_50
#SBATCH -p l40s
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --time=9999:59:59
#SBATCH -o ./logs/260502/%x.o%j
#SBATCH -e ./logs/260502/%x.e%j

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ligand-screen

_D="${SLURM_SUBMIT_DIR:-.}"
if   [ -d "$_D/src" ];      then ROOT="$(cd "$_D"    && pwd)"
elif [ -d "$_D/../src" ];   then ROOT="$(cd "$_D/.." && pwd)"
else                              ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi

export PYTHONPATH=$ROOT
cd "$ROOT"

#python -u  scripts/hpo_mlp.py \
#    --config outputs/runs/fp_100_cv1_opt/config_resolved.yaml \
#    --n_trials 200

python -u  scripts/hpo_mlp.py \
    --config configs/experiments/relu_50_not4.yaml \
    --csv data/ml_data_full.csv \
    --keep_neg_mid 100 --keep_neg 50 --keep_low_pos 50 \
    --n_trials 100 \
    --latent_dim 0
