#!/bin/bash
#SBATCH --job-name="HRM_Cycle_Cls"
#SBATCH -q acc_training
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=02:00:00
#SBATCH --output=logs/cycle_cls_%j.out
#SBATCH --error=logs/cycle_cls_%j.err
#SBATCH --chdir=.

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"

mkdir -p logs

module purge
module load miniforge
module load cuda/12.6
source activate deepLearning

echo "Raw activation baseline: linear classifier"
python probing/classify_cycle_embeddings.py \
  --data_dir probing_data_h_layer3 \
  --layer_filter H_level \
  --representation raw \
  --epochs 10

echo "UAE embedding: linear classifier"
python probing/classify_cycle_embeddings.py \
  --data_dir probing_data_h_layer3 \
  --layer_filter H_level \
  --representation uae \
  --weights results/h_uae/uae_weights.pt \
  --epochs 10

echo "SAE features: linear classifier"
python probing/classify_cycle_embeddings.py \
  --data_dir probing_data_h_layer3 \
  --layer_filter H_level \
  --representation sae \
  --weights results/h_sae/sae_weights.pt \
  --epochs 10

echo "Job finished at $(date)"
