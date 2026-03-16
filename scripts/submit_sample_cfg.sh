#!/bin/bash
#SBATCH -J sample_cfg
#SBATCH -p l40s
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --time=9999:59:59
#SBATCH -o ./logs/%x.o%j
#SBATCH -e ./logs/%x.e%j

# Usage:
#   sbatch scripts/submit_sample_cfg.sh sme_random 3.0 8.0
#   bash   scripts/submit_sample_cfg.sh sme_random 3.0 8.0   (로컬 테스트)
#
# Args:
#   $1  EXP             실험명 (default: sme_random)
#   $2  GUIDANCE_SCALE  CFG weight w (default: 3.0)
#   $3  TARGET_PIC50    목표 pIC50 (default: train 90th percentile)

EXP="${1:-sme_random}"
GUIDANCE_SCALE="${2:-3.0}"
TARGET_PIC50="${3:-}"

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
RUN_DIR="$ROOT/outputs/runs/${EXP}"
RESOLVED="$RUN_DIR/config_resolved.yaml"

mkdir -p logs "$RUN_DIR"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ligand_screen

export PYTHONPATH=$ROOT
cd "$ROOT"

# run_root를 실험 폴더로 고정
sed "s|run_root: \"auto\"|run_root: \"$RUN_DIR\"|" \
  "$ROOT/configs/experiments/${EXP}.yaml" > "$RESOLVED"

echo "[sample_cfg] EXP=$EXP  guidance_scale=$GUIDANCE_SCALE  target_pIC50=${TARGET_PIC50:-auto}"

# target_pic50 인수는 지정된 경우에만 추가
EXTRA_ARGS=""
if [ -n "$TARGET_PIC50" ]; then
  EXTRA_ARGS="--target_pic50 $TARGET_PIC50"
fi

$PYTHON "$ROOT/scripts/sample_cfg.py" \
  --config "$RESOLVED" \
  --guidance_scale "$GUIDANCE_SCALE" \
  --n_samples 500 \
  --top_k 50 \
  $EXTRA_ARGS
