#!/bin/bash
#SBATCH --job-name="HRM_Sudoku_Data"
#SBATCH -q acc_training
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=02:00:00
#SBATCH --output=logs/sudoku_data_%j.out
#SBATCH --error=logs/sudoku_data_%j.err
#SBATCH --chdir=.

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"

mkdir -p logs sudoku_probe_data_h_layer3

module purge
module load miniforge
module load cuda/12.6
source activate deepLearning

python probing/probing_sudoku_structured.py \
  --checkpoint "checkpoints/sudoku-extreme/checkpoint" \
  --batch_size 16 \
  --max_batches 50 \
  --output_dir "sudoku_probe_data_h_layer3" \
  --target_layers "model.inner.H_level.layers.3" \
  --return_logits

echo "Job finished at $(date)"
