#!/bin/bash
#SBATCH -J XXX_JOB
#SBATCH -p a5k2
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=9999:59:59
#SBATCH -o ./logs/260519/XXX_JOB.o%j
#SBATCH -e ./logs/260519/XXX_JOB.e%j

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

EXP="XXX_EXP"
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
RUN_DIR="$ROOT/outputs/runs/${EXP}"
RESOLVED="$RUN_DIR/config_resolved.yaml"

mkdir -p "$ROOT/logs" "$RUN_DIR"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ligand-screen

export PYTHONPATH=$ROOT
cd "$ROOT"

sed "s|run_root: \"auto\"|run_root: \"$RUN_DIR\"|" \
  "$ROOT/configs/experiments/${EXP}.yaml" > "$RESOLVED"

echo "[train_diffusion] EXP=$EXP  epochs=XXX_EP  T=XXX_T"

python "$ROOT/scripts/train_diffusion.py" \
  --config "$RESOLVED" \
  --diff_epochs XXX_EP \
  --T XXX_T
