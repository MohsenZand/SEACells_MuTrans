#!/bin/bash
# SLURM batch job for the LARRY SEACells + MuTrans pipeline (LARRY/scripts/supp_larry.py).
# Submit from the repository root:  sbatch LARRY/scripts/supp_larry.sbatch
#
# Edit the account/partition and environment activation to match your cluster.
# The SEACells fit is dominated by a large dense matmul in the Frank-Wolfe update
# (1200x1200 @ 1200x92527); the thread env vars below let BLAS parallelize it
# across the allocated cores. Peak memory observed ~80 GB.

#SBATCH --job-name=larry_seacells
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=36:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

set -euo pipefail

# --- Multi-threaded BLAS: use all allocated cores for the dense matmuls ---
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# --- Activate your Python environment (edit to match your setup) ---
# conda activate ipsc_mutrans_seacells
# Point MUTRANS_PATH at your MuTrans checkout and (optionally) the raw data:
# export MUTRANS_PATH=/path/to/MuTrans-release
# export LARRY_RAW_H5AD=/path/to/larry_NM_trajectory_normedCounts.h5ad

# Run from the repository root
cd "${SLURM_SUBMIT_DIR:-.}"
python -u LARRY/scripts/supp_larry.py
