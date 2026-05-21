#!/bin/bash
#SBATCH -J qubit_ex
#SBATCH -p l40s
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=9999:59:59
#SBATCH -o ./logs/260511/%x.o%j
#SBATCH -e ./logs/260511/%x.e%j

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ligand-screen

_D="${SLURM_SUBMIT_DIR:-.}"
if   [ -d "$_D/src" ];      then ROOT="$(cd "$_D"    && pwd)"
elif [ -d "$_D/../src" ];   then ROOT="$(cd "$_D/.." && pwd)"
else                              ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi

export PYTHONPATH=$ROOT
cd "$ROOT"

#python -m src.utils.expressibility \
#    --mode data_dependent --vqc_type angle_reupload \
#    --latent_path outputs/runs/vqc_tan_sub_1/latents_train.npy \
#    --ckpt_path outputs/runs/vqc_tan_sub_1/diffusion/2026-05-08/T1000_ep3000/denoiser_cfg.pt \
#    --block_idx 0 \
#    --data_mode fixed_theta \
#    --n_qubits 4 --n_layers 2 --n_samples 5000 \
#    --outdir outputs/expr/fixed_trained_theta_bl_0 \

#python -m src.utils.analyze_vqc_encoding \
#    --latent_path outputs/runs/vqc_tan_sub_1/latents_train.npy \
#    --y_path      outputs/runs/vqc_tan_sub_1/y_train.npy \
#    --ckpt_path   outputs/runs/rgcn_vqc_z4_sub_1/diffusion/2026-04-24/T1000_ep3000/denoiser_cfg.pt \
#    --n_qubits 4 --n_layers 2 --block_idx 0 \
#    --n_samples 99999 \
#    --outdir outputs/anal_qubit/vqc_encoding_anal_no_tanh \
#    --no_tanh

#python -m src.utils.analyze_vqc_contribution \
#    --latent_path outputs/runs/vqc_tan_sub_1/latents_train.npy \
#    --y_path      outputs/runs/vqc_tan_sub_1/y_train.npy \
#    --ckpt_path   outputs/runs/vqc_tan_sub_1/diffusion/2026-05-08/T1000_ep3000/denoiser_cfg.pt \
#    --outdir outputs/vqc_contribution_analysis/vqc_tanh

python -m src.utils.analyze_vqc_contribution \
    --latent_path outputs/runs/rgcn_vqc_z4_sub_1/latents_train.npy \
    --y_path      outputs/runs/rgcn_vqc_z4_sub_1/y_train.npy \
    --ckpt_path   outputs/runs/rgcn_vqc_z4_sub_1/diffusion/2026-04-24/T1000_ep3000/denoiser_cfg.pt \
    --outdir outputs/vqc_contribution_analysis/vqc_no_tanh

#python -m src.utils.verify_qout_vs_sv \
#    --latent_path outputs/runs/vqc_tan_sub_1/latents_train.npy \
#    --y_path      outputs/runs/vqc_tan_sub_1/y_train.npy \
#    --ckpt_path   outputs/runs/vqc_tan_sub_1/diffusion/2026-05-08/T1000_ep3000/denoiser_cfg.pt \
#    --blocks 0,5
