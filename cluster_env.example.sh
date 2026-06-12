#!/bin/bash
# Template for cluster / SSH settings. Copy to cluster_env.local.sh and fill in.
#
#   cp cluster_env.example.sh cluster_env.local.sh
#
# cluster_env.local.sh is gitignored and never committed. Source it before
# submitting jobs or syncing code so your account / host stay out of git:
#
#   source cluster_env.local.sh
#   sbatch -A "$SLURM_ACCOUNT" sbatch_train_h_autoencoders.sh
#   ./sync_to_cluster.sh

# SLURM account/project code to charge jobs to (passed via `sbatch -A "$SLURM_ACCOUNT"`).
export SLURM_ACCOUNT="your_account_code"

# SSH login node and user for copying code/results to the cluster.
export CLUSTER_USER="your_username"
export CLUSTER_HOST="login.example.edu"
export CLUSTER_DIR="~/Probing-HRM"
