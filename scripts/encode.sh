#!/bin/bash
#SBATCH -J eval
#SBATCH -p l40s
#SBATCH -N 1
#SBATCH -n 8
##SBATCH --gres=gpu:1
#SBATCH --time=9999:59:59
#SBATCH -o ./logs/260428/%x.o%j
#SBATCH -e ./logs/260428/%x.e%j

mkdir -p logs

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ligand-screen

_D="${SLURM_SUBMIT_DIR:-.}"
if   [ -d "$_D/src" ];      then ROOT="$(cd "$_D"    && pwd)"
elif [ -d "$_D/../src" ];   then ROOT="$(cd "$_D/.." && pwd)"
else                              ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi

export PYTHONPATH=$ROOT
cd "$ROOT"

python scripts/eval_encoder_topk.py \
    --screening_preds screening/sub_4/preds_screening.npy \
    --screening_pic50 screening/sub_4/y_screening.npy \
    --screening_name sub_4 \
    --top_k 3 5 10 20 50 \
    --threshold 8.0

