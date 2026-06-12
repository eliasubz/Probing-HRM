#!/bin/bash
#SBATCH --job-name="HRM_Sudoku_Probe"
#SBATCH -q acc_training
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=02:00:00
#SBATCH --output=logs/sudoku_probe_%j.out
#SBATCH --error=logs/sudoku_probe_%j.err
#SBATCH --chdir=.

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"

mkdir -p logs results/sudoku_probe

module purge
module load miniforge
module load cuda/12.6
source activate deepLearning

echo "Linear digit probe across cycles and H inner calls..."
python probing/train_sudoku_probe.py \
  --data_dir sudoku_probe_data_h_layer3 \
  --output_csv results/sudoku_probe/h_layer3_linear_digit_probe.csv \
  --layer_name model_inner_H_level_layers_3 \
  --train_on all \
  --hidden_dim 0 \
  --epochs 10

echo "MLP digit probe across cycles and H inner calls..."
python probing/train_sudoku_probe.py \
  --data_dir sudoku_probe_data_h_layer3 \
  --output_csv results/sudoku_probe/h_layer3_mlp_digit_probe.csv \
  --layer_name model_inner_H_level_layers_3 \
  --train_on all \
  --hidden_dim 256 \
  --epochs 10

echo "Job finished at $(date)"
