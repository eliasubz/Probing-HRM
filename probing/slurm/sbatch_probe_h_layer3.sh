#!/bin/bash
#SBATCH --job-name="HRM_Probe_H_L3"
#SBATCH -q acc_training
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=02:00:00
#SBATCH --output=logs/probe_h_l3_%j.out
#SBATCH --error=logs/probe_h_l3_%j.err
#SBATCH --chdir=.

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"

mkdir -p logs

module purge
module load miniforge
module load cuda/12.6
source activate deepLearning

echo "Running final H-level layer probing..."
python probing/probing_analysis.py \
    --checkpoint "checkpoints/sudoku-extreme/checkpoint" \
    --batch_size 16 \
    --max_batches 50 \
    --output_dir "probing_data_h_layer3" \
    --target_layers "model.inner.H_level.layers.3"

echo "Job finished at $(date)"
