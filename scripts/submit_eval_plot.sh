#!/bin/bash
#SBATCH -J eval_plot_45_cos
#SBATCH -p l40s
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --time=9999:59:59
#SBATCH -o ./logs/260406/%x.o%j
#SBATCH -e ./logs/260406/%x.e%j

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

python scripts/evaluate_plot.py --run_dir outputs/runs/rgcn_vqc_z4_45 \
    --load_reducer outputs/runs/rgcn_mlp_z4_45/umap_reducer50_08_cos.pkl --umap_metric cosine
