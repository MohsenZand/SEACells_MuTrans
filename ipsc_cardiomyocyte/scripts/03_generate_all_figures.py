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
FIG_DIR = config.RESULTS_DIR / '03_all_figures'
FIG_DIR.mkdir(parents=True, exist_ok=True) 
sc.settings.figdir = FIG_DIR



def plot_image_panel(ax, file_path, label_char):
    """Helper to plot a pre-generated image onto an axis."""
    if file_path and file_path.exists():
        img = Image.open(file_path)
        ax.imshow(img)
    else:
        ax.text(0.5, 0.5, f'Image not found\n{label_char}', ha='center', va='center', transform=ax.transAxes)
    ax.axis('off')
    ax.set_title(label_char, fontsize=14, fontweight='bold', loc='left', pad=5)

def create_comprehensive_figure(adata_org, adata_seacell, transition_files, save_path, si1, sf1, si2, sf2, cell_level_heatmaps=False):
    """
    Main function to assemble the multi-panel figure.
    """
    print(f"Creating comprehensive figure: {save_path.name}")
    fig = plt.figure(figsize=(30, 12))
    gs = gridspec.GridSpec(3, 12, figure=fig, hspace=0.35, wspace=0.4,
                          left=0.04, right=0.98, top=0.96, bottom=0.04,
                          height_ratios=[1, 1.2, 1])

    # --- ROW 1: Cell-level UMAPs and Violins (always from adata_org) ---
    ax_a = fig.add_subplot(gs[0, 0:2]);  pl.plot_umap(adata_org, ax_a, 'entropy', 'A')
    ax_b = fig.add_subplot(gs[0, 2:4]);  pl.plot_umap(adata_org, ax_b, 'land', 'B')
    ax_c = fig.add_subplot(gs[0, 4:6]);  pl.plot_violin(adata_org, ax_c, 'entropy', 'C')
    ax_d = fig.add_subplot(gs[0, 6:8]);  pl.plot_violin(adata_org, ax_d, 'land', 'D')
    ax_e = fig.add_subplot(gs[0, 8:12]); pl.plot_flow_matrix(adata_org, ax_e, 'E')

    # --- ROW 2: Transition Plots (from adata_seacell) ---
    ax_f = fig.add_subplot(gs[1, 0:4]);  plot_image_panel(ax_f, transition_files.get('mpft'), 'F')
    ax_g = fig.add_subplot(gs[1, 4:8]);  plot_image_panel(ax_g, transition_files.get(f'{si1}to{sf1}'), 'G')
    ax_h = fig.add_subplot(gs[1, 8:12]); plot_image_panel(ax_h, transition_files.get(f'{si2}to{sf2}'), 'H')

    # --- ROW 3: Transcendental Heatmaps ---
    ax_i = fig.add_subplot(gs[2, 0:6])
    ax_j = fig.add_subplot(gs[2, 6:12])
    
    if cell_level_heatmaps:
        # Map memberships if they aren't already
        if 'rho_class' not in adata_org.obsm:
            adata_org = tran.map_seacell_memberships_to_cells(adata_org, adata_seacell)
        
        pl.plot_transcendental_heatmap(
            adata_org, si1, sf1, ax_i, max_cells=5000, 
            region_method='adaptive', title=f'I: Transition A{si1} → A{sf1} (Cells)',
            fig_dir=FIG_DIR
        )
        pl.plot_transcendental_heatmap(
            adata_org, si2, sf2, ax_j, max_cells=5000, 
            region_method='adaptive', title=f'J: Transition A{si2} → A{sf2} (Cells)',
            fig_dir=FIG_DIR
        )
    else: # Metacell-level heatmaps
        pl.plot_transcendental_heatmap(
            adata_seacell, si1, sf1, ax_i, 
            region_method='logistic', title=f'I: Transition A{si1} → A{sf1} (Metacells)',
            fig_dir=FIG_DIR
        )
        pl.plot_transcendental_heatmap(
            adata_seacell, si2, sf2, ax_j, 
            region_method='logistic', title=f'J: Transition A{si2} → A{sf2} (Metacells)',
            fig_dir=FIG_DIR
        )
    
    # --- Save Figure ---
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ...Saved {save_path.name}")

def main():
    print("\n" + "="*80)
    print("Generating All Final Figures")
    print("="*80)
    
    # 1. Load data
    print("Loading data...")
    adata_org = sc.read(config.MUTRANS_AD_ORG)
    adata_seacell = sc.read(config.MUTRANS_AD_SEACELL)

    # 2. Generate transition images (published transitions: A10->A9 and A10->A4)
    print("Generating transition plots...")
    transition_files = {}
    si1, sf1 = '10', '9'
    transition_files = pl.generate_mutrans_transition_plots(adata_seacell, FIG_DIR, si1, sf1, transition_files)
    si2, sf2 = '10', '4'
    transition_files = pl.generate_mutrans_transition_plots(adata_seacell, FIG_DIR, si2, sf2, transition_files)

    # 3. Create Metacell (SEACell) version of the figure -- the published one
    save_path_seacells = FIG_DIR / f'Fig_Comprehensive_Metacells_{si1}_{sf1}_and_{si2}_{sf2}.pdf'
    create_comprehensive_figure(
        adata_org, adata_seacell, transition_files,
        save_path_seacells, si1, sf1, si2, sf2, cell_level_heatmaps=False,

    )

    # 4. Single-cell version -- DISABLED: only metacell-level TD genes are published.
    #    (Masked out cell-level TD identification/plot; keep for reference only.)
    # save_path_cells = FIG_DIR / f'Fig_Comprehensive_Cells_{si1}_{sf1}_and_{si2}_{sf2}.pdf'
    # create_comprehensive_figure(
    #     adata_org, adata_seacell, transition_files,
    #     save_path_cells, si1, sf1, si2, sf2, cell_level_heatmaps=True,
    # )

    print("\n" + "="*80)
    print("All figures generated!")
    print("="*80)

if __name__ == "__main__":
    main()