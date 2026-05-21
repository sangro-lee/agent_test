#!/bin/bash
#SBATCH -J GA_1
#SBATCH -p l40s
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=9999:59:59
#SBATCH -o ../logs/260518/%x.o%j
#SBATCH -e ../logs/260518/%x.e%j

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ligand-screen

_D="${SLURM_SUBMIT_DIR:-.}"
if   [ -d "$_D/src" ];      then ROOT="$(cd "$_D"    && pwd)"
elif [ -d "$_D/../src" ];   then ROOT="$(cd "$_D/.." && pwd)"
elif [ -d "$_D/../../src" ];   then ROOT="$(cd "$_D/../.." && pwd)"
else                              ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi

export PYTHONPATH=$ROOT
cd "$ROOT"

python scripts/r2gb/genetic_latent.py \
    --config outputs/runs/vqc_tan_sub_1/config_resolved.yaml \
    --initpool data/initpool_bace1_sub1.dat --smiles_col smiles \
    --mode latent \
    --z_samples outputs/runs/vqc_tan_sub_1/diffusion/2026-05-08/T1000_ep3000/cfg_w1.0_ddim/z_samples.npy \
    --mutate_rxn scripts/r2gb/mutate_reaction.dat \
    --nstep 50 \
    --n_candidates 20 \
    --out_dir outputs/ga_latent/vqc_tanh_w1

