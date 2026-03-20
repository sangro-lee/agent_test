#!/bin/bash
# Run train_diffusion.py for multiple experiment configs.
#
# Usage:
#   bash scripts/run_diffusion_multi.sh [--epochs N] [--T N] EXP1 EXP2 ...
#
# Examples:
#   bash scripts/run_diffusion_multi.sh sme_random sme_unet_random sme_unet_vqc_random
#   bash scripts/run_diffusion_multi.sh --epochs 500 --T 500 sme_unet_random gnn_unet_random
#   bash scripts/run_diffusion_multi.sh --epochs 300 sme_random sme_unet_random sme_unet_vqc_random gnn_unet_random gnn_unet_vqc_random

DIFF_EPOCHS=200
T=1000
EXPS=()

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs) DIFF_EPOCHS="$2"; shift 2 ;;
        --T)      T="$2";           shift 2 ;;
        *)        EXPS+=("$1");     shift   ;;
    esac
done

if [[ ${#EXPS[@]} -eq 0 ]]; then
    echo "Usage: $0 [--epochs N] [--T N] EXP1 EXP2 ..."
    echo "Example: $0 --epochs 300 --T 500 sme_random sme_unet_random"
    exit 1
fi

# ── Root 탐지 ─────────────────────────────────────────────────────────────────
_D="${SLURM_SUBMIT_DIR:-.}"
if   [ -d "$_D/src" ];    then ROOT="$(cd "$_D"    && pwd)"
elif [ -d "$_D/../src" ]; then ROOT="$(cd "$_D/.." && pwd)"
else                           ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi

PYTHON="${PYTHON:-python}"
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ligand-screen 2>/dev/null || true

export PYTHONPATH="$ROOT"
cd "$ROOT"
mkdir -p logs

echo "=========================================="
echo " diff_epochs=${DIFF_EPOCHS}  T=${T}"
echo " experiments: ${EXPS[*]}"
echo "=========================================="

# ── 실험 순차 실행 ────────────────────────────────────────────────────────────
FAILED=()
for EXP in "${EXPS[@]}"; do
    RUN_DIR="$ROOT/outputs/runs/${EXP}"
    RESOLVED="$RUN_DIR/config_resolved.yaml"
    mkdir -p "$RUN_DIR"

    sed "s|run_root: \"auto\"|run_root: \"$RUN_DIR\"|" \
        "$ROOT/configs/experiments/${EXP}.yaml" > "$RESOLVED"

    echo ""
    echo "── [$EXP] 시작 ──"
    $PYTHON "$ROOT/scripts/train_diffusion.py" \
        --config "$RESOLVED" \
        --diff_epochs "$DIFF_EPOCHS" \
        --T "$T"

    if [[ $? -ne 0 ]]; then
        echo "!! [$EXP] 실패"
        FAILED+=("$EXP")
    else
        echo "── [$EXP] 완료 ──"
    fi
done

# ── 결과 요약 ─────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo " 완료: $((${#EXPS[@]} - ${#FAILED[@]})) / ${#EXPS[@]}"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo " 실패: ${FAILED[*]}"
fi
echo "=========================================="
