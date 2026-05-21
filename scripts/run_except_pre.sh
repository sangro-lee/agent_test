#!/bin/bash
# 6개 실험 순차 실행: (mlp|gnn|sme) x (scaffold|random)

# 스크립트 위치 기준으로 ROOT 자동 감지 (어느 서버에서든 동작)
_D="${SLURM_SUBMIT_DIR:-.}"
if   [ -d "$_D/src" ];      then ROOT="$(cd "$_D"    && pwd)"
elif [ -d "$_D/../src" ];   then ROOT="$(cd "$_D/.." && pwd)"
else                              ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi
PYTHON="${PYTHON:-python}"
CONFIGS=$ROOT/configs/experiments

export PYTHONPATH=$ROOT
cd $ROOT

EXPERIMENTS=(
#"vqc_un_100_norm_cv0"
#"vqc_un_100_norm_cv1"
#"vqc_un_100_norm_cv2"
#"vqc_un_100_norm_cv3"
#"vqc_un_100_norm_cv4"
#"vqc_mur_un_100_cv0"
#"vqc_none_100_cv4_no_scale"
#"vqc_none_100_cv4"
"fp_50_fin"
#"ortho_fp_50_cv1"
#"vqc_mur_un_100_cv1"
#"vqc_mur_un_100_cv2"
#"vqc_mur_un_100_cv3"
#"vqc_mur_un_50_cv0"
#"vqc_mur_un_50_cv1"
#"vqc_mur_un_50_cv2"
#"vqc_mur_un_50_cv3"
#"vqc_mur_un_50_cv4"
#"vqc_ps_norm_cv0"
#"vqc_ps_norm_cv1"
#"vqc_ps_norm_cv2"
#"vqc_ps_norm_cv3"
#"vqc_ps_norm_cv4"
#"vqc_z4_ps_cv0"
#"vqc_z4_ps_cv1"
#"vqc_z4_ps_cv2"
#"vqc_z4_ps_cv3"
#"vqc_z4_ps_cv4"
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
  mkdir -p "$RUN_DIR"
  RESOLVED_CFG="$RUN_DIR/config_resolved.yaml"
  sed "s|run_root: \"auto\"|run_root: \"$RUN_DIR\"|" \
    $CONFIGS/${EXP}.yaml > $RESOLVED_CFG

  echo "--- train ---"
  if ! $PYTHON $ROOT/scripts/train.py --config $RESOLVED_CFG 2>&1; then
    echo "[$EXP] train FAILED"
    FAILED+=("$EXP")
    continue
  fi

  echo "--- evaluate ---"
  if $PYTHON $ROOT/scripts/evaluate.py --config $RESOLVED_CFG 2>&1; then
    echo "[$EXP] DONE"
  else
    echo "[$EXP] evaluate FAILED"
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
