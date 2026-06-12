#!/bin/bash
#SBATCH --job-name="HRM_Probe_Layer3"
#SBATCH -q acc_training
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=02:00:00
#SBATCH --output=logs/probe_layer3_%j.out
#SBATCH --error=logs/probe_layer3_%j.err
#SBATCH --chdir=.

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"

mkdir -p logs

module purge
module load miniforge
module load cuda/12.6
source activate deepLearning

echo "Running Layer 3 Probing..."
python probing/probing_analysis.py \
    --checkpoint "checkpoints/sudoku-extreme/checkpoint" \
    --batch_size 16 \
    --max_batches 5 \
    --output_dir "probing_data_layer3" \
    --target_layers "model.inner.H_level.layers.3" "model.inner.L_level.layers.3"

echo "Job finished at $(date)"
