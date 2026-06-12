#!/bin/bash
#SBATCH --job-name="HRM_Train_UAE"
#SBATCH -q acc_training
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=02:00:00
#SBATCH --output=logs/train_uae_%j.out
#SBATCH --error=logs/train_uae_%j.err
#SBATCH --chdir=.

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"

mkdir -p logs
mkdir -p results/uae

module purge
module load miniforge
module load cuda/12.6
source activate deepLearning

echo "Training Undercomplete Autoencoder..."
python probing/train_uae_cluster.py \
  --data_dir probing_data_layer3 \
  --output_dir results/uae

echo "Job finished at $(date)"
