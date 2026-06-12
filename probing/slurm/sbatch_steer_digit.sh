#!/bin/bash
#SBATCH --job-name="HRM_Steer_Digit"
#SBATCH -q acc_training
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=02:00:00
#SBATCH --output=logs/steer_digit_%j.out
#SBATCH --error=logs/steer_digit_%j.err
#SBATCH --chdir=.

# Activation steering with a digit-specific SAE feature.
# Prerequisite: run sbatch_analyze_sae_cells.sh first to produce
#   results/cell_analysis/top_digit_specific_features.csv
#
# STEER_DIGIT: which digit (1-9) to pick the top feature for.
# Submit with: STEER_DIGIT=5 CKPT=checkpoints/.../step_XXXX \
#                sbatch -A "$SLURM_ACCOUNT" probing/slurm/sbatch_steer_digit.sh

CKPT="${CKPT:-checkpoints/your_project/your_run/step_XXXX}"
STEER_DIGIT="${STEER_DIGIT:-5}"

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Steering toward digit: $STEER_DIGIT"

mkdir -p logs results/steering results/steering_random results/figures

module purge
module load miniforge
module load cuda/12.6
source activate deepLearning

# Real digit-specific SAE feature direction
python probing/steer_activations.py \
  --checkpoint "$CKPT" \
  --vector_source sae \
  --sae_weights results/h_sae/sae_weights.pt \
  --digit_features_csv results/cell_analysis/top_digit_specific_features.csv \
  --steer_digit "$STEER_DIGIT" \
  --target_layer model.inner.H_level.layers.0 \
  --inject_steps all \
  --alphas -8 -4 -2 0 2 4 8 \
  --output_dir "results/steering_digit${STEER_DIGIT}"

# Random control (same norm, no semantic direction)
python probing/steer_activations.py \
  --checkpoint "$CKPT" \
  --vector_source random \
  --target_layer model.inner.H_level.layers.0 \
  --inject_steps all \
  --alphas -8 -4 -2 0 2 4 8 \
  --output_dir results/steering_random

echo "Job finished at $(date)"
