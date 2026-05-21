#!/bin/bash
#SBATCH -J eval
#SBATCH -p fast48
#SBATCH -N 1
#SBATCH -n 8
##SBATCH --gres=gpu:1
#SBATCH --time=9999:59:59
#SBATCH -o ./logs/260410/%x.o%j
#SBATCH -e ./logs/260410/%x.e%j

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

# 1. MLP 기준으로 fit하고 reducer 저장 (1회만)
python scripts/plot_latents.py --run_dir outputs/runs/rgcn_mlp_z4_47 \
    --no_tsne --umap_n_neighbors 20 --umap_min_dist 0.8 \
    --save_reducer outputs/runs/rgcn_mlp_z4_45/umap_reducer20_08_eu.pkl --no_tsne --umap_metric euclidean


# 2. 이후 다른 실험은 저장된 reducer로 transform (re-fit 없음, 빠름)
#python scripts/plot_latents.py --run_dir outputs/runs/rgcn_ortho_z4 \
#    --load_reducer outputs/runs/rgcn_mlp_z4/umap_reducer07.pkl --no_tsne

