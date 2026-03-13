#!/bin/bash
# 6개 실험 순차 실행: (mlp|gnn|sme) x (scaffold|random)

# 스크립트 위치 기준으로 ROOT 자동 감지 (어느 서버에서든 동작)
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"
CONFIGS=$ROOT/configs/experiments

export PYTHONPATH=$ROOT
cd $ROOT

EXPERIMENTS=(
  "mlp_scaffold"
  "mlp_random"
  "gnn_scaffold"
  "gnn_random"
  "sme_scaffold"
  "sme_random"
)

TOTAL=${#EXPERIMENTS[@]}
FAILED=()

for i in "${!EXPERIMENTS[@]}"; do
  EXP="${EXPERIMENTS[$i]}"
  RUN_DIR="$ROOT/outputs/runs/${EXP}"
  echo ""
  echo "========================================"
  echo " [$((i+1))/$TOTAL] $EXP"
  echo "========================================"

  # output.run_root를 실험명으로 고정 (타임스탬프 충돌 방지)
  RESOLVED_CFG="/tmp/${EXP}_resolved.yaml"
  sed "s|run_root: \"auto\"|run_root: \"$RUN_DIR\"|" \
    $CONFIGS/${EXP}.yaml > $RESOLVED_CFG

  echo "--- preprocess ---"
  if ! $PYTHON $ROOT/scripts/preprocess.py --config $RESOLVED_CFG 2>&1; then
    echo "[$EXP] preprocess FAILED"
    FAILED+=("$EXP")
    continue
  fi

  echo "--- train ---"
  if $PYTHON $ROOT/scripts/train.py --config $RESOLVED_CFG 2>&1; then
    echo "[$EXP] DONE"
  else
    echo "[$EXP] train FAILED"
    FAILED+=("$EXP")
  fi
done

echo ""
echo "========================================"
echo "All $TOTAL experiments finished."
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "FAILED: ${FAILED[*]}"
else
  echo "All succeeded."
fi
