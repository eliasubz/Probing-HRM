#!/bin/bash
# Push changed scripts to the cluster. Reads connection details from the
# gitignored cluster_env.local.sh (see cluster_env.example.sh) so no
# usernames or hosts are committed.
set -euo pipefail

if [ ! -f cluster_env.local.sh ]; then
  echo "cluster_env.local.sh not found. Copy cluster_env.example.sh first." >&2
  exit 1
fi
source cluster_env.local.sh

# Mirror the probing/ tree (scripts + slurm wrappers) to the cluster.
scp -r probing "${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_DIR}/"
