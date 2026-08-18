#!/usr/bin/env python3
"""
Step 03: Comprehensive Figure Generation
Generates and assembles all panels for the main figures
(both single-cell and metacell versions).
"""
import sys
import os
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import seaborn as sns
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src import config
from src import plotting as pl
from src import transcendental as tran

sc.settings.verbosity = 0
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 8

sns.set_style('ticks')
matplotlib.rcParams['figure.figsize'] = [4, 4]
matplotlib.rcParams['figure.dpi'] = 100
FIG_DIR = config.RESULTS_DIR / '04_heatmaps'
FIG_DIR.mkdir(parents=True, exist_ok=True) 
sc.settings.figdir = FIG_DIR






def main():
    parser = argparse.ArgumentParser(description='Generating Heatmap Figures')
    parser.add_argument('--si', type=str, default='3', help='')
    parser.add_argument('--sf', type=str, default='4', help='')
    parser.add_argument('--original_clusters', type=bool, default=False, help='')
    parser.add_argument('--cluster_name', type=str, default='leiden_0.5', help='')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print(f"Generating Heatmap Figures for {args.si} to {args.sf}")
    print("="*80)
    
    # 1. Load data
    print("Loading data...")
    adata_org = sc.read(config.MUTRANS_AD_ORG)
    adata_seacell = sc.read(config.MUTRANS_AD_SEACELL)

    # 2. Generate transition images 
    print("Generating transition plots...")
    transition_files = {}
    
    si, sf = args.si, args.sf
    cluster_name = args.cluster_name
    
    fig = plt.figure(figsize=(30, 12))
    
    # Cell-level heatmaps
    if 'rho_class' not in adata_org.obsm:
            adata_org = tran.map_seacell_memberships_to_cells(adata_org, adata_seacell)
        
    if args.original_clusters:
        gs = gridspec.GridSpec(1, 1, figure=fig, hspace=0.35, wspace=0.4,
                            left=0.04, right=0.98, top=0.96, bottom=0.04,
                            height_ratios=[1])
        
        ax_i = fig.add_subplot(gs[0, 0])
    
        pl.plot_transcendental_heatmap_cell(
                adata_org, si, sf, ax_i, max_cells=5000, 
                region_method='adaptive', title=f'Transition C{si} → C{sf} (Cells) -- Clustered by {cluster_name}',
                fig_dir=FIG_DIR,
                cluster_name=cluster_name
            )
        
        save_path = FIG_DIR / f'Heatmap_{si}_{sf}_original_clusters.pdf'

    else:
        gs = gridspec.GridSpec(1, 2, figure=fig, hspace=0.35, wspace=0.4,
                            left=0.04, right=0.98, top=0.96, bottom=0.04,
                            height_ratios=[1])
        
        ax_i = fig.add_subplot(gs[0, 0])
        ax_j = fig.add_subplot(gs[0, 1])
    
        pl.plot_transcendental_heatmap(
                adata_org, si, sf, ax_i, max_cells=5000, 
                region_method='adaptive', title=f'I: Transition A{si} → A{sf} (Cells)',
                fig_dir=FIG_DIR
            )
        # Metacell-level heatmaps
        pl.plot_transcendental_heatmap(
            adata_seacell, si, sf, ax_j, 
            region_method='logistic', title=f'I: Transition A{si} → A{sf} (Metacells)',
            fig_dir=FIG_DIR
        )
        
        save_path = FIG_DIR / f'Heatmap_{si}_{sf}.pdf'
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ...Saved {save_path.name}")
    
    
    
    print("\n" + "="*80)
    print("All figures generated!")
    print("="*80)

if __name__ == "__main__":
    main()