#!/bin/bash
#SBATCH -J protein_test   # Job Name (XXX를 seed로 대체)
#SBATCH -p l40s                  # Partition
#SBATCH -N 1                        # Node count
#SBATCH -n 8                        # CPU cores
#SBATCH --gres=gpu:1                # GPU 1개 할당
#SBATCH --time=9999:59:59           # Time limit
#SBATCH -o ./logs/%x.o%j                   # stdout 파일
#SBATCH -e ./logs/%x.e%j                   # stderr 파일

source /home/sangro/anaconda3/etc/profile.d/conda.sh
conda activate ligand-screen
bash ./run_all_experiments.sh
