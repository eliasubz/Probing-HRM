#!/bin/bash
#SBATCH --job-name="HRM_SAE_Cycles"
#SBATCH -q acc_training
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=01:00:00
#SBATCH --output=logs/sae_cycles_%j.out
#SBATCH --error=logs/sae_cycles_%j.err
#SBATCH --chdir=.

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"

mkdir -p logs results/cycle_analysis

module purge
module load miniforge
module load cuda/12.6
source activate deepLearning

python probing/analyze_sae_cycles.py \
  --data_dir probing_data_h_layer3 \
  --weights results/h_sae/sae_weights.pt \
  --output_dir results/cycle_analysis \
  --layer_filter H_level \
  --top_k 50

echo "Job finished at $(date)"
