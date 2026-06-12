#!/bin/bash
#SBATCH --job-name="MAI_PW2_Latent_Probing"
#SBATCH -q acc_training
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=02:00:00
#SBATCH --output=logs/probing_%j.out
#SBATCH --error=logs/probing_%j.err
#SBATCH --chdir=.

# SLURM Job Script for HRM Probing
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"

# Create logs directory if it doesn't exist
mkdir -p logs

# 1. Setup Environment (Load the optimized BSC PyTorch)
module purge
module load miniforge
module load cuda/12.6
source activate deepLearning

# Install our offline dependencies locally
echo "Ensuring offline dependencies are installed..."
python -m pip install --user --no-index --find-links=bsc_wheels einops coolname pydantic argdantic wandb omegaconf hydra-core huggingface_hub PyYAML


# 2. Run the script
python probing/probing_analysis.py \
    --checkpoint "checkpoints/sudoku-extreme/checkpoint" \
    --batch_size 16 \
    --max_batches 50 \
    --output_dir "probing_data" \
    --target_layers "model.inner.H_level.layers.0" "model.inner.L_level.layers.0"

echo "Job finished at $(date)"
