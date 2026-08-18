#!/usr/bin/env python3
"""
Step 02: Run MuTrans Analysis
Loads SEACell summaries and computes dynamical metrics.
Saves .h5ad files with MuTrans results.
"""
import sys
import scanpy as sc
import pandas as pd
import numpy as np
import collections.abc
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

# Add project root to path to allow 'from src import ...'
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src import config  # Import paths and params
sys.path.append(config.MUTRANS_PATH)
from MuTrans.Example import pyMuTrans as pm

sns.set_style('ticks')
matplotlib.rcParams['figure.figsize'] = [4, 4]
matplotlib.rcParams['figure.dpi'] = 100
RESULTS_DIR = config.RESULTS_DIR / '02_mutrans_analysis'
FIG_DIR = RESULTS_DIR / 'figures'
RESULTS_DIR.mkdir(parents=True, exist_ok=True) 
FIG_DIR.mkdir(parents=True, exist_ok=True) 
sc.settings.figdir = FIG_DIR


# --- Helper Functions (specific to this script) ---

def find_and_convert_matlab_objects(obj, key_path=None):
    """(Your function, unchanged)"""
    if key_path is None: key_path = []
    if isinstance(obj, collections.abc.Mapping):
        keys_to_convert = []
        for k, v in obj.items():
            current_path = key_path + [k]
            if isinstance(v, collections.abc.Mapping):
                find_and_convert_matlab_objects(v, key_path=current_path)
            elif 'matlab' in str(type(v)):
                keys_to_convert.append(k)
        for k in keys_to_convert:
            v = obj[k]
            print(f"Converting '{k}' at {' -> '.join(key_path + [k])} to numpy array.")
            try: obj[k] = np.asarray(v)
            except Exception: obj[k] = str(v)

def load_data():
    print("\n[1/5] Loading data files...")
    adata_org_seacell = sc.read(config.SEACELL_AD_ORG)
    adata_seacell = sc.read(config.SEACELL_AD_SUMMARY)
    print(f"  - SEACells summary: {adata_seacell.shape}")
    print(f"  - Original with SEACells: {adata_org_seacell.shape}")
    return adata_org_seacell, adata_seacell

def preprocess_and_cluster(adata_seacell):
    print("\n[2/5] Preprocessing and clustering...")
    print(f"  Raw layer max: {adata_seacell.layers['raw'].toarray().max()}")
    print(f"  Raw layer min: {adata_seacell.layers['raw'].toarray().min()}")
    print(f"  X max: {adata_seacell.X.toarray().max()}")
    print(f"  X min: {adata_seacell.X.toarray().min()}")
    
    # Quality control filtering
    print(f"  Shape before filtering: {adata_seacell.shape}")
    sc.pp.filter_cells(adata_seacell, min_genes=300)
    print(f"  After cell filtering: {adata_seacell.shape}")
    sc.pp.filter_genes(adata_seacell, min_cells=10)
    print(f"  After gene filtering: {adata_seacell.shape}")
    
    # Identify gene categories
    adata_seacell.var["mt"] = adata_seacell.var_names.str.startswith("MT-")
    adata_seacell.var["ribo"] = adata_seacell.var_names.str.startswith(("RPS", "RPL"))
    adata_seacell.var["hb"] = adata_seacell.var_names.str.contains("^HB[^(P)]")
    
    # Calculate QC metrics
    sc.pp.calculate_qc_metrics(adata_seacell, qc_vars=["mt", "ribo", "hb"], 
                              inplace=True, log1p=True)
    
    # Plot QC violin plots
    sc.pl.violin(adata_seacell,
                ["n_genes_by_counts", "total_counts", "pct_counts_mt", 
                 "pct_counts_ribo", "pct_counts_hb"],
                jitter=0.4,
                multi_panel=True,
                save='_qc_leiden_0.5.pdf')
    
    print("\n Normalization and clustering...")
    
    # Normalization
    sc.pp.normalize_total(adata_seacell, target_sum=1e4)
    sc.pp.log1p(adata_seacell)
    
    # Highly variable genes
    sc.pp.highly_variable_genes(adata_seacell, n_top_genes=3000)
    adata_seacell = adata_seacell[:, adata_seacell.var.highly_variable]
    print(f"  After HVG selection: {adata_seacell.shape}")
    
    # Dimensionality reduction and clustering
    sc.pp.pca(adata_seacell)
    sc.pp.neighbors(adata_seacell)
    sc.tl.umap(adata_seacell)
    sc.tl.leiden(adata_seacell, resolution=0.5, key_added='leiden_0.5')
    
    # Plot UMAP
    sc.pl.umap(adata_seacell, color=['leiden_0.5'], legend_loc='on data',
              save='_leiden_0.5.pdf')
    
    
    return adata_seacell

def run_mutrans(adata_seacell):
    print("\n[3/5] Running MuTrans dynamical analysis...")
    if not os.path.exists(config.MUTRANS_AD_SEACELL):
        par = {
            "choice_distance": "cosine",
            "perplex": config.MUTRAMS_PERPLEX,
            "K_cluster": config.MUTRAMS_K_CLUSTER,
            "reduction_coord": 'umap',
            "weight_scale": False,
            "write_anndata": True,
            "learn_regulatory_network": True,
            "generate_interactive_vis": True,
            "reduce_large_scale": False,
            "force_double_precision": True,
        }
        adata_seacell = pm.dynamical_analysis(adata_seacell, par)
        return adata_seacell
    else:
        adata_seacell = sc.read(config.MUTRANS_AD_SEACELL)

def transfer_and_save(adata_seacell, adata_org_seacell):
    print("\n[4/5] Transferring metrics and saving...")
    if not os.path.exists(config.MUTRANS_AD_SEACELL):
        # Transfer metrics
        ent_map = adata_seacell.obs['entropy'].to_dict()
        adata_org_seacell.obs['entropy'] = adata_org_seacell.obs['SEACell'].map(ent_map)
        for key in ['land', 'attractor']:
            if key in adata_seacell.obs:
                m = adata_seacell.obs[key].to_dict()
                adata_org_seacell.obs[key] = adata_org_seacell.obs['SEACell'].map(m)

        # Convert and save
        find_and_convert_matlab_objects(adata_org_seacell.uns)
        adata_org_seacell.write(config.MUTRANS_AD_ORG)
        print(f"  Saved: {config.MUTRANS_AD_ORG.name}")
        
        find_and_convert_matlab_objects(adata_seacell.uns)
        adata_seacell.write(config.MUTRANS_AD_SEACELL)
        print(f"  Saved: {config.MUTRANS_AD_SEACELL.name}")
    else:
        adata_org_seacell = sc.read(config.MUTRANS_AD_ORG)
        adata_seacell = sc.read(config.MUTRANS_AD_SEACELL)
        
    return adata_seacell, adata_org_seacell


def create_violin_plots(adata_seacell, adata_org_seacell):
    """Create violin plots for entropy and land metrics"""
    
    # Violin plots by leiden cluster (org)
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    sc.pl.violin(adata_org_seacell, keys=['entropy'], groupby='leiden_0.5',
                jitter=False, rotation=45, show=False, ax=axs[0], xlabel='leiden 0.5')
    sc.pl.violin(adata_org_seacell, keys=['land'], groupby='leiden_0.5',
                jitter=False, rotation=45, show=False, ax=axs[1], xlabel='leiden 0.5')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'violin_org_leiden.pdf')
    plt.close()
    
    # Violin plots by leiden cluster (seacell)
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    sc.pl.violin(adata_seacell, keys=['entropy'], groupby='leiden_0.5',
                jitter=False, rotation=45, show=False, ax=axs[0], xlabel='leiden 0.5')
    sc.pl.violin(adata_seacell, keys=['land'], groupby='leiden_0.5',
                jitter=False, rotation=45, show=False, ax=axs[1], xlabel='leiden 0.5')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'violin_seacell_leiden.pdf')
    plt.close()
    
    # Violin plots by attractor (org)
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    sc.pl.violin(adata_org_seacell, keys=['entropy'], groupby='attractor',
                jitter=False, rotation=45, show=False, ax=axs[0], xlabel='attractor')
    sc.pl.violin(adata_org_seacell, keys=['land'], groupby='attractor',
                jitter=False, rotation=45, show=False, ax=axs[1], xlabel='attractor')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'violin_org_attractor.pdf')
    plt.close()
    
    # Violin plots by attractor (seacell)
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    sc.pl.violin(adata_seacell, keys=['entropy'], groupby='attractor',
                jitter=False, rotation=45, show=False, ax=axs[0], xlabel='attractor')
    sc.pl.violin(adata_seacell, keys=['land'], groupby='attractor',
                jitter=False, rotation=45, show=False, ax=axs[1], xlabel='attractor')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'violin_seacell_attractor.pdf')
    plt.close()
    
        
    
def visualize_lineage(adata_seacell, adata_org_seacell):
    print("\n[5/5] Visualizing the lineage network ...")

    # Ensure proper data types
    adata_org_seacell.obs['entropy'] = adata_org_seacell.obs['entropy'].astype(float)
    
    '''
    # Main UMAP with multiple features
    sc.pl.umap(adata_org_seacell,
              color=['leiden_0.5', 'entropy', 'attractor', 'land'],
              save='_attractor_entropy_land_leiden_0.5.pdf',
              ncols=4,
              legend_loc='on data')
    
    # Plot original SEACells UMAP
    sc.pl.umap(adata_org_seacell, color=['leiden_0.5'], legend_loc='on data',
             save='_org_leiden_0.5.pdf')
    
    
    # Rank genes by attractor
    sc.tl.rank_genes_groups(adata_org_seacell, groupby="attractor", method="wilcoxon")
    
    sc.pl.rank_genes_groups_heatmap(
        adata_org_seacell,
        groupby="attractor",
        n_genes=5,
        show=False,
        vmin=-1, vmax=3,
        dendrogram=True,
        swap_axes=True,
        show_gene_labels=True,
        figsize=(10, 6),
        save='_org_rank_genes_groups_leiden_0.5.pdf')
    
    sc.tl.rank_genes_groups(adata_org_seacell, groupby="attractor", 
                           method="t-test", n_genes=20)
    
    sc.pl.rank_genes_groups(adata_org_seacell, n_genes=20, sharey=False,
                           save='_rank_genes_attractor_leiden_0.5.pdf')
    
    # Marker genes dotplot
    marker_genes = [
        'POU5F1', 'SOX2', 'ESRG',  # Pluripotent stem cell
        'MIXL1',  # Primitive streak
        'TUBB2B', 'TUBB2A', 'CNTNAP2',
        'CRABP2', 'APLNR', 'HAS2',  # Mesodermal progenitors
        'MEIS2', 'RPL7P9',
        'KRT19', 'TTR',  # Endoderm
        'ALB', 'APOB', 'APOC3',  # Liver
        'APOA1', 'FN1',
        'LIX1', 'MEIS1', 'PDGFRA',  # Cardiac progenitor
        'ISL1', 'HAND1', 'MAB21L2',
        'GATA4', 'TBX5', 'NKX2-5',
        'TNNT2', 'MYL7', 'TTN', 'MYH6',  # Cardiomyocyte
        'TAGLN', 'CALD1', 'ACTA2',  # Muscle
        'COL3A1', 'COL1A2', 'LUM'  # Cardiac fibroblast
    ]
    
    sc.pl.dotplot(
        adata_org_seacell,
        var_names=marker_genes,
        groupby='attractor',
        standard_scale='var',
        dot_max=0.8,
        dot_min=0.05,
        color_map='Reds',
        dendrogram=True,
        show=False,
        save='_markers_leiden_0.5.pdf')
    '''
    
    
    # Violin plots for entropy and land
    create_violin_plots(adata_seacell, adata_org_seacell)
    
    
    # MPFT lineage inference
    fig = plt.figure(figsize=(10, 6))
    pm.infer_lineage(adata_seacell, method="MPFT", flux_fraction=0.3,
                    size_point=40, alpha_point=0.5, size_text=20, color_palette=None)
    plt.savefig(FIG_DIR / 'mpft_transition_leiden_0.5.pdf')
    plt.close()
    
    # Extract transition matrix
    transition_matrix = adata_seacell.uns['land']['land']
    N_land = transition_matrix.shape[0]
    
    land_ids = adata_seacell.obs_names
    if len(land_ids) != N_land:
        print(f"  Warning: land_ids length ({len(land_ids)}) != matrix size ({N_land})")
        land_ids = land_ids[:N_land]
    
    # Convert matrix to edge list
    SCORE_THRESHOLD = 1e-6
    source_nodes, target_nodes, scores = [], [], []
    
    for i in range(N_land):
        for j in range(N_land):
            score = transition_matrix[i, j]
            if score > SCORE_THRESHOLD:
                source_nodes.append(land_ids[i])
                target_nodes.append(land_ids[j])
                scores.append(score)
    
    network_df = pd.DataFrame({
        'Source_LAND': source_nodes,
        'Target_LAND': target_nodes,
        'Linkage_Score': scores
    })
    
    network_df.to_csv(RESULTS_DIR / 'land_lineage_network_table.csv', index=False)
    print(f"  Extracted {len(network_df)} linkages from {N_land}x{N_land} matrix")
    
    # Aggregate by attractor
    ATTRACTOR_COL = 'attractor'
    seacell_to_attractor_map = adata_seacell.obs[ATTRACTOR_COL].astype(str)
    
    network_df['Source_Attractor'] = network_df['Source_LAND'].map(seacell_to_attractor_map)
    network_df['Target_Attractor'] = network_df['Target_LAND'].map(seacell_to_attractor_map)
    
    attractor_network_df = network_df.groupby(
        ['Source_Attractor', 'Target_Attractor']
    )['Linkage_Score'].sum().reset_index()
    
    attractor_network_df.rename(columns={
        'Source_Attractor': 'Source_Attractor_ID',
        'Target_Attractor': 'Target_Attractor_ID',
        'Linkage_Score': 'Total_Linkage_Score'
    }, inplace=True)
    
    attractor_network_df.to_csv(RESULTS_DIR / 'attractor_lineage_network_aggregated.csv', 
                                index=False)
    print(f"  Aggregated to {len(attractor_network_df)} attractor-to-attractor linkages")
    
    # Analyze specific attractor (e.g., attractor 4)
    TARGET_ATTRACTOR = '4'
    attractor_4_links = attractor_network_df[
        attractor_network_df['Source_Attractor_ID'] == TARGET_ATTRACTOR
    ].sort_values(by='Total_Linkage_Score', ascending=False)
    
    print(f"\n  Links FROM Attractor #{TARGET_ATTRACTOR}:")
    print(attractor_4_links)
    
    # MPPT lineage inference (specific paths)
    fig = plt.figure(figsize=(10, 6))
    pm.infer_lineage(adata_seacell, si=10, sf=4, method="MPPT",
                    flux_fraction=0.3, size_point=40, alpha_point=0.5,
                    size_text=15, color_palette=None)
    plt.savefig(FIG_DIR / 'mppt_transition_10_4_leiden_0.5.pdf')
    plt.close()
    
    fig = plt.figure(figsize=(10, 6))
    pm.infer_lineage(adata_seacell, si=3, sf=4, method="MPPT",
                    flux_fraction=0.3, size_point=40, alpha_point=0.5,
                    size_text=15, color_palette=None)
    plt.savefig(FIG_DIR / 'mppt_transition_3_4_leiden_0.5.pdf')
    plt.close()
    
    print("  Lineage network analysis complete")
    

def main():
    print("=" * 80)
    print("MuTrans Analysis Pipeline")
    print("=" * 80)
    adata_org_seacell, adata_seacell = load_data()
    adata_seacell = preprocess_and_cluster(adata_seacell)
    adata_seacell = run_mutrans(adata_seacell)
    adata_seacell, adata_org_seacell = transfer_and_save(adata_seacell, adata_org_seacell)
    visualize_lineage(adata_seacell, adata_org_seacell)
    print("\n" + "=" * 80)
    print("Analysis complete! Processed .h5ad files saved.")
    print("=" * 80)

if __name__ == "__main__":
    main()