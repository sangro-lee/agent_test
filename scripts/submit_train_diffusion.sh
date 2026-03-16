#!/bin/bash
#SBATCH -J train_diffusion
#SBATCH -p l40s
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --time=9999:59:59
#SBATCH -o ./logs/%x.o%j
#SBATCH -e ./logs/%x.e%j

# Usage:
#   sbatch scripts/submit_train_diffusion.sh sme_random
#   bash   scripts/submit_train_diffusion.sh sme_random   (로컬 테스트)

EXP="${1:-sme_random}"
_D="${SLURM_SUBMIT_DIR:-.}"
if   [ -d "$_D/src" ];      then ROOT="$(cd "$_D"    && pwd)"
elif [ -d "$_D/../src" ];   then ROOT="$(cd "$_D/.." && pwd)"
else                              ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi
PYTHON="${PYTHON:-python}"
RUN_DIR="$ROOT/outputs/runs/${EXP}"
RESOLVED="$RUN_DIR/config_resolved.yaml"

mkdir -p logs "$RUN_DIR"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ligand-screen

export PYTHONPATH=$ROOT
cd "$ROOT"

# run_root를 실험 폴더로 고정 (auto → 실제 경로)
sed "s|run_root: \"auto\"|run_root: \"$RUN_DIR\"|" \
  "$ROOT/configs/experiments/${EXP}.yaml" > "$RESOLVED"

echo "[train_diffusion] EXP=$EXP"
echo "[train_diffusion] config=$RESOLVED"

$PYTHON "$ROOT/scripts/train_diffusion.py" \
  --config "$RESOLVED" \
  --diff_epochs 200 \
  --T 1000
