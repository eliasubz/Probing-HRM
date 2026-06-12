#!/bin/bash
#SBATCH --job-name="HRM_Train_H_AE"
#SBATCH -q acc_training
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_h_ae_%j.out
#SBATCH --error=logs/train_h_ae_%j.err
#SBATCH --chdir=.

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"

mkdir -p logs results/h_uae results/h_sae

module purge
module load miniforge
module load cuda/12.6
source activate deepLearning

echo "Training H-level undercomplete autoencoder..."
python probing/train_uae_cluster.py \
  --data_dir probing_data_h_layer3 \
  --output_dir results/h_uae \
  --layer_filter H_level

echo "Training H-level sparse autoencoder..."
python probing/train_sae_cluster.py \
  --data_dir probing_data_h_layer3 \
  --output_dir results/h_sae \
  --layer_filter H_level \
  --expansion_factor 4 \
  --l1_coeff 1e-3

echo "Job finished at $(date)"
