#!/bin/bash
#SBATCH -J eval_45_cos
#SBATCH -p fast48
#SBATCH -N 1
#SBATCH -n 8
##SBATCH --gres=gpu:1
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


#python scripts/plot_umap_sweep.py --run_dir outputs/runs/rgcn_mlp_z4_45 \
#    --n_neighbors 10 20 30 40 50 60 70 80 90 100 \
#    --min_dist 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9

python scripts/plot_umap_sweep.py --run_dir outputs/runs/rgcn_mlp_z4_45 \
    --n_neighbors 10 20 30 40 50 60 70 80 90 100\
    --min_dist 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9\
    --umap_metric cosine
