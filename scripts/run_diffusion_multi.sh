#!/bin/bash
# Submit train_diffusion jobs for all combinations of EXPs × epochs × T.
# Edit the arrays below, then: bash scripts/run_diffusion_multi.sh

EXPS=(
#"vqc_sub_1_0519_2"
#"vqc_sub_1_0520_nf"
#"mlp_sub_1_0520_nf"
"mlp_sub_1_0520_nf_tan"
#"mlp_sub_1_0519_nf"
)
EPOCHS=(3000 5000)
TS=(1000 2000 3000)

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TEMPLATE="$ROOT/scripts/submit_train_diffusion.sh"
mkdir -p "$ROOT/logs"
cd "$ROOT"   # sbatch는 ROOT에서 실행 → SLURM_SUBMIT_DIR=ROOT

for EXP in "${EXPS[@]}"; do
    for EP in "${EPOCHS[@]}"; do
        for T in "${TS[@]}"; do
            JOB="${EXP}__ep${EP}__T${T}"
            tmp="tmp_${JOB}.sh"
            sed -e "s/XXX_JOB/${JOB}/g" \
                -e "s/XXX_EXP/${EXP}/g" \
                -e "s/XXX_EP/${EP}/g"   \
                -e "s/XXX_T/${T}/g"     \
                "$TEMPLATE" > "$tmp"
            sbatch "$tmp"
            rm -f "$tmp"
        done
    done
done
