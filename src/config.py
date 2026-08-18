"""Central configuration: paths and parameters shared across the pipeline.

All paths are repo-relative by default. Machine-specific inputs (the raw data
file and the external MuTrans install) are read from environment variables with
placeholder defaults, so nothing here is tied to a particular machine:

    export IPSC_RAW_H5AD=/path/to/GSE175634_iPSC_CM.sct3k_reclustered.h5ad
    export MUTRANS_PATH=/path/to/MuTrans-release      # https://github.com/cliffzhou92/MuTrans-release

See the top-level README for setup details.
"""
import os
from pathlib import Path

# --- Project root (repo root; this file lives in <root>/src/) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Parameters ---
LEIDEN_RESOLUTION = 0.5
N_TOP_HVG = 3000
SEACELL_N_WAYPOINT_EIGS = 10
SEACELL_CONVERGENCE_EPS = 1e-5
MUTRAMS_PERPLEX = 200.0
MUTRAMS_K_CLUSTER = 14

# --- External dependencies (override via environment variables) ---
# MuTrans is not on PyPI; clone it and point MUTRANS_PATH at the checkout.
MUTRANS_PATH = os.environ.get('MUTRANS_PATH', '/path/to/MuTrans-release')

# --- Input data ---
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
# Download GSE175634 from GEO and set IPSC_RAW_H5AD, or place the file under data/raw/.
RAW_H5AD = os.environ.get('IPSC_RAW_H5AD',
                          str(RAW_DATA_DIR / 'GSE175634_iPSC_CM.sct3k_reclustered.h5ad'))

# --- Processed data paths ---
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
SEACELL_DIR = PROCESSED_DATA_DIR / 'seacells'
SEACELL_AD_ORG = SEACELL_DIR / 'output_with_SEACells.h5ad'
SEACELL_AD_SUMMARY = SEACELL_DIR / 'SEACell_summary.h5ad'
SEACELL_AD_SOFT = SEACELL_DIR / 'SEACell_soft_summary.h5ad'

MUTRANS_DIR = PROCESSED_DATA_DIR / 'mutrans'
MUTRANS_AD_SEACELL = MUTRANS_DIR / f'seacells_mutrans_{LEIDEN_RESOLUTION}.h5ad'
MUTRANS_AD_ORG = MUTRANS_DIR / f'seacells_org_mutrans_{LEIDEN_RESOLUTION}.h5ad'

# --- Results ---
RESULTS_DIR = PROJECT_ROOT / 'results'

# Ensure output directories exist
DIRS_TO_MAKE = [
    SEACELL_DIR, MUTRANS_DIR, RESULTS_DIR,
    RESULTS_DIR / '01_seacell_analysis',
    RESULTS_DIR / '02_mutrans_analysis',
    RESULTS_DIR / '03_all_figures',
]
for d in DIRS_TO_MAKE:
    d.mkdir(parents=True, exist_ok=True)
