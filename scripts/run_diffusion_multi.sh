#!/bin/bash
# Run train_diffusion.py for all combinations of EXPs × epochs × T.
#
# Usage:
#   bash scripts/run_diffusion_multi.sh \
#       --exps  "sme_random sme_unet_random sme_unet_vqc_random" \
#       --epochs "200 500" \
#       --T     "500 1000"
#
# 위 예시: 3 × 2 × 2 = 12개 조합 순차 실행
# 각 조합의 출력 폴더: outputs/runs/<EXP>/diffusion/T<T>_ep<epochs>/

EXPS_STR=""
EPOCHS_STR="200"
T_STR="1000"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --exps)   EXPS_STR="$2";   shift 2 ;;
        --epochs) EPOCHS_STR="$2"; shift 2 ;;
        --T)      T_STR="$2";      shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

read -ra EXPS   <<< "$EXPS_STR"
read -ra EPOCHS <<< "$EPOCHS_STR"
read -ra TS     <<< "$T_STR"

if [[ ${#EXPS[@]} -eq 0 ]]; then
    echo "Usage: $0 --exps \"EXP1 EXP2\" [--epochs \"200 500\"] [--T \"500 1000\"]"
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

TOTAL=$(( ${#EXPS[@]} * ${#EPOCHS[@]} * ${#TS[@]} ))
echo "=========================================="
echo " 조합: ${#EXPS[@]} exps × ${#EPOCHS[@]} epochs × ${#TS[@]} T = ${TOTAL}개"
echo " exps:   ${EXPS[*]}"
echo " epochs: ${EPOCHS[*]}"
echo " T:      ${TS[*]}"
echo "=========================================="

# ── 전체 조합 순차 실행 ───────────────────────────────────────────────────────
FAILED=()
IDX=0
for EXP in "${EXPS[@]}"; do
    RUN_DIR="$ROOT/outputs/runs/${EXP}"
    RESOLVED="$RUN_DIR/config_resolved.yaml"
    mkdir -p "$RUN_DIR"

    sed "s|run_root: \"auto\"|run_root: \"$RUN_DIR\"|" \
        "$ROOT/configs/experiments/${EXP}.yaml" > "$RESOLVED"

    for EP in "${EPOCHS[@]}"; do
        for T in "${TS[@]}"; do
            IDX=$(( IDX + 1 ))
            JOB="${EXP}__ep${EP}__T${T}"
            echo ""
            echo "── [${IDX}/${TOTAL}] ${JOB} 시작 ──"

            $PYTHON "$ROOT/scripts/train_diffusion.py" \
                --config "$RESOLVED" \
                --diff_epochs "$EP" \
                --T "$T" \
                2>&1 | tee "logs/${JOB}.log"

            if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
                echo "!! [${JOB}] 실패"
                FAILED+=("$JOB")
            else
                echo "── [${JOB}] 완료 ──"
            fi
        done
    done
done

# ── 결과 요약 ─────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo " 완료: $(( TOTAL - ${#FAILED[@]} )) / ${TOTAL}"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo " 실패: ${FAILED[*]}"
fi
echo "=========================================="
