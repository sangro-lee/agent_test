#!/bin/bash
#SBATCH -J ligand_screen
#SBATCH -p l40s
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --time=9999:59:59
#SBATCH -o ./logs/%x.o%j
#SBATCH -e ./logs/%x.e%j

mkdir -p logs

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ligand_screen

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
bash "$ROOT/scripts/run_all_experiments.sh"
