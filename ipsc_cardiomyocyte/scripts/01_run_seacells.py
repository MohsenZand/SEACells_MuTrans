#!/usr/bin/env python3
"""
Step 01: SEACell Computation
Loads raw .h5ad data, computes SEACells (metacells),
and saves the SEACell-summarized AnnData objects.
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import SEACells  
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Add project root to path to allow 'from src import ...'
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src import config # Import paths and params

# Set random seeds for reproducibility
np.random.seed(0)
random.seed(0)

# Configure plotting
sns.set_style('ticks')
matplotlib.rcParams['figure.figsize'] = [4, 4]
matplotlib.rcParams['figure.dpi'] = 100
FIG_DIR = config.RESULTS_DIR / '01_seacell_analysis'
FIG_DIR.mkdir(parents=True, exist_ok=True)
sc.settings.figdir = FIG_DIR

def plot_and_save(fig_name):
    """Helper to save plots to the correct results directory."""
    plt.tight_layout()
    plt.savefig(FIG_DIR / fig_name)
    plt.close()
    
def initialize_data(filepath):
    """Loads and pre-processes the raw AnnData object."""
    print(f"Loading data from: {filepath}")
    ad = sc.read(filepath)
    print(f'Original data shape: {ad.shape}')
    
    # Use 'data' layer 
    ad.X = ad.layers['data']
    ad.X = ad.X.astype(np.float32)
    sc.pp.pca(ad, n_comps=30)
    
    # Map time
    diffdays_mapping = {
        'day0': 0, 'day1': 1, 'day3': 3,
        'day5': 5, 'day7': 7, 'day11': 11, 'day15': 15
    }
    ad.obs["time"] = ad.obs["diffday"].map(diffdays_mapping).astype(int)
    
    # Correct the raw count assignment
    print(f'Raw count shape: {ad.raw.X.shape}')
    if not ad.raw.var_names.is_unique:
        temp_raw_ad = sc.AnnData(ad.raw.X, var=ad.raw.var)
        temp_raw_ad.var_names_make_unique()
        ad.raw = temp_raw_ad
        print("Made adata.raw.var_names unique.")
    
    ad.raw.obs = ad.obs
    return ad


def run_seacells(ad, n_SEACells, build_kernel_on='X_pca', n_waypoint_eigs=10):
    """Initializes and fits the SEACells model."""
    print(f"Running SEACells with n_SEACells = {n_SEACells}")
    model = SEACells.core.SEACells(
        ad,
        build_kernel_on=build_kernel_on,
        n_SEACells=n_SEACells,
        n_waypoint_eigs=config.SEACELL_N_WAYPOINT_EIGS,
        convergence_epsilon=config.SEACELL_CONVERGENCE_EPS
    )
    model.construct_kernel_matrix()
    
    # Plot kernel matrix
    try:
        sns.clustermap(model.kernel_matrix[:500, :500].toarray())
        plot_and_save("kernel_matrix_clustermap.png")
    except Exception as e:
        print(f"Could not plot kernel matrix: {e}")

    model.initialize_archetypes()
    SEACells.plot.plot_initialization(ad, model, save_as=str(FIG_DIR / "initialization_umap.png"))

    print("Fitting model...")
    model.fit(min_iter=10, max_iter=50)
    
    model.plot_convergence(save_as=str(FIG_DIR / "rss_convergence.png"))
    return model

def summarize_and_evaluate(ad, model):
    """Generates plots for the SEACells model."""
    print("Evaluating model and plotting ...")
    
    # (saving to FIG_DIR)
    SEACells.plot.plot_2D(ad, key='X_umap', colour_metacells=False, save_as=str(FIG_DIR / "umap_cells.png"))
    SEACells.plot.plot_2D(ad, key='X_umap', colour_metacells=True, save_as=str(FIG_DIR / "umap_metacells.png"))
    SEACells.plot.plot_SEACell_sizes(ad, bins=5, save_as=str(FIG_DIR / "seacell_sizes.png"))

    # Purity
    purity = SEACells.evaluate.compute_celltype_purity(ad, 'leiden_0.5')
    plt.figure(figsize=(4, 4)); sns.boxplot(data=purity, y='leiden_0.5_purity'); plt.title('Celltype Purity'); sns.despine()
    plot_and_save("celltype_purity.png")
    
    # Compactness
    compactness = SEACells.evaluate.compactness(ad, 'X_pca')
    plt.figure(figsize=(4, 4)); sns.boxplot(data=compactness, y='compactness'); plt.title('Compactness'); sns.despine()
    plot_and_save("compactness.png")

    # Separation
    separation = SEACells.evaluate.separation(ad, 'X_pca', nth_nbr=1)
    plt.figure(figsize=(4, 4)); sns.boxplot(data=separation, y='separation'); plt.title('Separation'); sns.despine()
    plot_and_save("separation.png")


def main():
    print("\n" + "="*80)
    print("Step 01: SEACell Computation Pipeline")
    print("="*80)
    
    # 1. Load data
    ad = initialize_data(config.RAW_H5AD)
    
    # 2. Save reference UMAPs
    sc.pl.scatter(ad, basis='umap', color='leiden_0.5_type', frameon=False, save="_type.png", show=False)
    sc.pl.scatter(ad, basis='umap', color='leiden_0.5', frameon=False, save="_leiden_0.5.png", show=False)
    
    # 3. Run SEACells
    n_cells = ad.shape[0]
    n_SEACells = n_cells // 200 
    model = run_seacells(ad, n_SEACells)

    # 4. Save results to data/processed/seacells/
    print("Saving processed data...")
    ad.obs['SEACell'] = model.get_hard_assignments()
    ad.write(config.SEACELL_AD_ORG)
    print(f"  Saved: {config.SEACELL_AD_ORG.name}")

    # 5. Evaluate model
    summarize_and_evaluate(ad, model)

    # 6. Save summarized versions
    SEACell_ad = SEACells.core.summarize_by_SEACell(ad, SEACells_label='SEACell', summarize_layer='raw', ad_raw_var_names=True)
    SEACell_ad.write(config.SEACELL_AD_SUMMARY)
    print(f"  Saved: {config.SEACELL_AD_SUMMARY.name}")

    SEACell_soft_ad = SEACells.core.summarize_by_soft_SEACell(ad, model.A_, celltype_label='leiden_0.5', summarize_layer='raw', 
                                                            minimum_weight=0.05, ad_raw_var_names=True)
    SEACell_soft_ad.write(config.SEACELL_AD_SOFT)
    print(f"  Saved: {config.SEACELL_AD_SOFT.name}")
    
    print("\n" + "="*80)
    print("SEACell computation complete!")
    print("="*80)

if __name__ == "__main__":
    main()