#!/bin/bash
#SBATCH --job-name="HRM_SAE_Cells"
#SBATCH -q acc_training
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=01:00:00
#SBATCH --output=logs/sae_cells_%j.out
#SBATCH --error=logs/sae_cells_%j.err
#SBATCH --chdir=.

# Find which SAE features are specific to individual Sudoku digits.
# Prerequisite: structured probe data must exist (run sbatch_probe_sudoku_structured.sh first).
# Submit with: sbatch -A "$SLURM_ACCOUNT" probing/slurm/sbatch_analyze_sae_cells.sh

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"

mkdir -p logs results/cell_analysis

module purge
module load miniforge
module load cuda/12.6
source activate deepLearning

# Full analysis: all cells (given + blank)
python probing/analyze_sae_cells.py \
  --data_dir sudoku_probe_data_h_layer3 \
  --sae_weights results/h_sae/sae_weights.pt \
  --output_dir results/cell_analysis \
  --act_cycle -1 \
  --inner_call -1 \
  --top_k 20

# Blank-only: only cells that were originally blank (harder, more interesting)
python probing/analyze_sae_cells.py \
  --data_dir sudoku_probe_data_h_layer3 \
  --sae_weights results/h_sae/sae_weights.pt \
  --output_dir results/cell_analysis_blanks \
  --act_cycle -1 \
  --inner_call -1 \
  --top_k 20 \
  --blanks_only

echo "Job finished at $(date)"
