#!/bin/bash
# Submit sample_cfg jobs for all combinations of EXPS × GUIDANCE_SCALES.
# Edit the arrays below, then: bash scripts/run_sample_cfg_multi.sh

# ── 설정 ───────────────────────────────────────────────────────────────────
EXPS=(
#"fix_mlp_z4_sub_1_xh"
#"mlp_sub_1_0519_nf"
"mlp_sub_1_0519"
#"vqc_sub_1_0519_nf"
#"vqc_sub_1_0519"
)
GUIDANCE_SCALES=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 1.0 1.5 2.0 2.5 3.0 3.5 4.0 5.0 5.5 6.0 6.5 7.0 10.0)
DATE="2026-05-19"           # diffusion 날짜 폴더 (e.g. "2026-03-31"); 비우면 legacy path
TARGET_PIC50=""   # 고정 목표 pIC50; 비우면 train 90th percentile 자동 사용
# ──────────────────────────────────────────────────────────────────────────

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TEMPLATE="$ROOT/scripts/submit_sample_cfg.sh"
mkdir -p "$ROOT/logs"
cd "$ROOT"

for EXP in "${EXPS[@]}"; do
    for W in "${GUIDANCE_SCALES[@]}"; do
        JOB="${EXP}__w${W}"
        tmp="tmp_${JOB}.sh"
        sed "s/XXX_JOB/${JOB}/g" "$TEMPLATE" > "$tmp"
        echo "[multi] sbatch $JOB  date=${DATE:-legacy}  target_pIC50=${TARGET_PIC50:-auto}"
        sbatch "$tmp" "$EXP" "$W" "$TARGET_PIC50" "$DATE"
        rm -f "$tmp"
    done
done
