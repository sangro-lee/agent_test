#!/bin/bash
# Submit sample_cfg jobs for all combinations of EXPS × GUIDANCE_SCALES.
# Edit the arrays below, then: bash scripts/run_sample_cfg_multi.sh

# ── 설정 ───────────────────────────────────────────────────────────────────
EXPS=("sme_random")
GUIDANCE_SCALES=(1.0 2.0 3.0 5.0)
DATE=""           # diffusion 날짜 폴더 (e.g. "2026-03-31"); 비우면 legacy path
TARGET_PIC50=""   # 고정 목표 pIC50; 비우면 train 90th percentile 자동 사용
# ──────────────────────────────────────────────────────────────────────────

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$ROOT/scripts/submit_sample_cfg.sh"
mkdir -p "$ROOT/logs"
cd "$ROOT"

for EXP in "${EXPS[@]}"; do
    for W in "${GUIDANCE_SCALES[@]}"; do
        echo "[multi] sbatch $EXP  w=$W  date=${DATE:-legacy}  target_pIC50=${TARGET_PIC50:-auto}"
        sbatch "$SCRIPT" "$EXP" "$W" "$TARGET_PIC50" "$DATE"
    done
done
