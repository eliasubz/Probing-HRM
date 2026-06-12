#!/bin/bash
#SBATCH --job-name="HRM_Steer_H_SAE"
#SBATCH -q acc_training
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=02:00:00
#SBATCH --output=logs/steer_h_sae_%j.out
#SBATCH --error=logs/steer_h_sae_%j.err
#SBATCH --chdir=.

# Activation steering with an SAE feature direction on the H-level residual
# stream. The feature is auto-selected as the most cycle-specific one from
# analyze_sae_cycles.py's CSV. Submit with your account:
#   sbatch -A "$SLURM_ACCOUNT" probing/slurm/sbatch_steer_h_sae.sh
#
# Set CKPT to your trained checkpoint (the dir must also contain all_config.yaml).
CKPT="${CKPT:-checkpoints/your_project/your_run/step_XXXX}"

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"

mkdir -p logs results/steering results/figures

module purge
module load miniforge
module load cuda/12.6
source activate deepLearning

# Real SAE feature direction (auto-picked cycle-specific feature)
python probing/steer_activations.py \
  --checkpoint "$CKPT" \
  --vector_source sae \
  --sae_weights results/h_sae/sae_weights.pt \
  --cycle_features_csv results/cycle_analysis/top_cycle_specific_features.csv \
  --target_layer model.inner.H_level.layers.0 \
  --inject_steps all \
  --alphas -8 -4 -2 0 2 4 8 \
  --output_dir results/steering

# Mandatory control: random direction of equal norm
python probing/steer_activations.py \
  --checkpoint "$CKPT" \
  --vector_source random \
  --target_layer model.inner.H_level.layers.0 \
  --inject_steps all \
  --alphas -8 -4 -2 0 2 4 8 \
  --output_dir results/steering_random

echo "Job finished at $(date)"
