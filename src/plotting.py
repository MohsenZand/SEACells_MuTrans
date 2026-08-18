"""
Central plotting library for the iPSC Cardiomyocyte project.
Contains all reusable plotting functions for UMAPs, violins,
flow matrices, and the transcendental analysis heatmap.
"""
import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import sys

# Ensure src is in path if this module is used in complex ways
# (though typically not needed if project is run from root)
from . import transcendental as tran
from . import config

# Import MuTrans, handling potential errors
try:
    sys.path.append(config.MUTRANS_PATH)
    from MuTrans.Example import pyMuTrans as pm
except ImportError:
    print("Error: pyMuTrans not found.")
    sys.exit(1)


def plot_umap(adata, ax, feature, label):
    """Plots a feature on a UMAP embedding."""
    if feature in adata.obs:
        if not pd.api.types.is_numeric_dtype(adata.obs[feature]):
            values = pd.to_numeric(adata.obs[feature], errors='coerce').values
        else:
            values = adata.obs[feature].values
    else:
        values = None
        
    if 'X_umap' not in adata.obsm or values is None:
        ax.text(0.5, 0.5, f'{feature} unavailable', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return
    
    umap = adata.obsm['X_umap']
    valid = ~np.isnan(values)
    
    sc_plot = ax.scatter(umap[valid, 0], umap[valid, 1], c=values[valid], 
                         cmap='viridis', s=0.8, alpha=0.7, rasterized=True, edgecolors='none')
    
    cbar = plt.colorbar(sc_plot, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(feature.capitalize(), rotation=270, labelpad=15, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    
    ax.set_xlabel('UMAP 1', fontsize=9)
    ax.set_ylabel('UMAP 2', fontsize=9)
    ax.set_title(label, fontsize=14, fontweight='bold', loc='left')
    ax.tick_params(labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_violin(adata, ax, feature, label, groupby='leiden_0.5'):
    """Creates a violin plot for a feature, grouped by a category."""
    if feature in adata.obs:
        if not pd.api.types.is_numeric_dtype(adata.obs[feature]):
            numeric_values = pd.to_numeric(adata.obs[feature], errors='coerce')
        else:
            numeric_values = adata.obs[feature]
    else:
        numeric_values = None
        
    if groupby not in adata.obs or numeric_values is None:
        ax.text(0.5, 0.5, 'Data unavailable', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return
    
    # Sample for performance
    n_sample = min(50000, len(adata))
    idx = np.random.choice(len(adata), n_sample, replace=False)
    adata_sample = adata[idx].copy()
    
    df = pd.DataFrame({
        'group': adata_sample.obs[groupby].astype(str),
        'value': numeric_values[adata_sample.obs_names]
    }).dropna()
    
    order = sorted(df['group'].unique(), key=lambda x: int(x) if x.isdigit() else x)
    palette = sns.color_palette("husl", len(order))
    
    sns.violinplot(data=df, x='group', y='value', order=order, ax=ax, 
                   inner='box', palette=palette, linewidth=1)
    
    ax.set_xlabel(groupby.replace('_', ' ').capitalize(), fontsize=9)
    ax.set_ylabel(feature.capitalize(), fontsize=9)
    ax.set_title(label, fontsize=14, fontweight='bold', loc='left')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_flow_matrix(adata, ax, label, groupby='leiden_0.5', target='attractor'):
    """Plots a heatmap of proportions between two categorical variables."""
    try:
        ct = pd.crosstab(
            adata.obs[groupby].astype(str), 
            adata.obs[target].astype(str), 
            normalize='index'
        )
        # Sort rows and columns numerically if possible
        ct = ct.reindex(sorted(ct.index, key=lambda x: int(x) if x.isdigit() else x))
        ct = ct.reindex(sorted(ct.columns, key=lambda x: int(x) if x.isdigit() else x), axis=1)

        sns.heatmap(
            ct, cmap='Reds', cbar_kws={'label': f'Proportion within {groupby} cluster'},
            ax=ax, linewidths=0.3
        )
        
        ax.set_xlabel(target.capitalize(), fontsize=9)
        ax.set_ylabel(groupby.replace('_', ' ').capitalize(), fontsize=9)
        ax.set_title(label, fontsize=14, fontweight='bold', loc='left')
        ax.tick_params(labelsize=8)
    except Exception as e:
        ax.text(0.5, 0.5, f'Flow Matrix error: {e}', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')


def generate_mutrans_transition_plots(adata_seacell, fig_dir, si, sf, transition_files):
    """
    Generates and saves all three MuTrans transition plots (MPFT, MPPTs).
    Returns a dictionary of file paths.
    """
    print(f"Generating MuTrans transition plots in {fig_dir}...")
    fig_dir.mkdir(parents=True, exist_ok=True)
    

    try:
        fig = plt.figure(figsize=(10, 8))
        pm.infer_lineage(adata_seacell, method="MPFT", flux_fraction=0.3,
                        size_point=40, alpha_point=0.5, size_text=20)
        path = fig_dir / 'transition_MPFT.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        transition_files['mpft'] = path
    except Exception as e:
        print(f"Error in generating MPFT figure: {e}")
        transition_files['mpft'] = None
    
    try:
        fig = plt.figure(figsize=(8, 6))
        pm.infer_lineage(adata_seacell, si=int(si), sf=int(sf), method="MPPT",
                        flux_fraction=0.3, size_point=40, alpha_point=0.5)
        path = fig_dir / f'transition_{si}to{sf}.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        transition_files[f"{si}to{sf}"] = path
    except Exception as e:
        print(f"Error in generating MPPT figure: {e}")
        transition_files[f"{si}to{sf}"] = None


    return transition_files


def plot_transcendental_heatmap(adata, source_attr, target_attr, ax,
                                n_genes_per_category=5, max_cells=None, 
                                region_method='logistic', title=None, fig_dir='.'):
    """
    Master function to create a transcendental analysis heatmap.
    
    This function calls src.transcendental helpers to get data and then focuses *only* on plotting.
    """
    adata.obs['attractor'] = adata.obs['attractor'].astype(str)
    mask = (adata.obs['attractor'] == source_attr) | (adata.obs['attractor'] == target_attr)
    
    if mask.sum() < 10:
        ax.text(0.5, 0.5, f'Insufficient data\nA{source_attr}→A{target_attr}',
               ha='center', va='center', transform=ax.transAxes); return

    adata_trans = adata[mask].copy()
    tcs = tran.calculate_tcs(adata_trans, source_attr, target_attr)
    sort_idx = np.argsort(tcs)[::-1] # Sort descending
    adata_trans = adata_trans[sort_idx].copy()
    tcs_sorted = tcs[sort_idx]
    
    if max_cells and len(tcs_sorted) > max_cells:
        sample_idx = np.linspace(0, len(tcs_sorted)-1, max_cells, dtype=int)
        adata_trans = adata_trans[sample_idx].copy()
        tcs_sorted = tcs_sorted[sample_idx]
    
    # 1. Get analysis from src.transcendental
    regions = tran.identify_transition_regions(tcs_sorted, method=region_method)
    stable_source, transition, stable_target = regions
    
    td_genes, td_genes_scores = tran.identify_td_genes(adata_trans, tcs_sorted, regions, n_genes=n_genes_per_category)
    
    columns = ['Genes', 'abs(corr)', 'corr', 'pval']
    pd_td_genes = pd.DataFrame(td_genes_scores, columns=columns)
    
    cell_seacell = 'seacell' if 'Metacell' in title else 'cell'
    csv_file_name = fig_dir / f"td_genes_scores_{source_attr}_to_{target_attr}_{cell_seacell}.csv"
    pd_td_genes.to_csv(csv_file_name, index=False) 
    print(f"\nTD Genes saved to {csv_file_name}.")
    
    ms_down, ms_up, ih_down, ih_up = tran.identify_ms_ih_genes(
        adata_trans, regions, td_genes, n_genes=n_genes_per_category
    )
    
    all_genes = ms_down + ih_down + td_genes + ih_up + ms_up
    gene_labels = (['MS↓'] * len(ms_down) + ['IH↓'] * len(ih_down) +
                   ['TD'] * len(td_genes) + ['IH↑'] * len(ih_up) + ['MS↑'] * len(ms_up))
    
    if len(all_genes) == 0:
        ax.text(0.5, 0.5, 'No significant genes found', ha='center', va='center', transform=ax.transAxes); return

    # 2. Prepare expression matrix for plotting
    all_genes = [g for g in all_genes if g in adata_trans.var_names]
    expr_matrix = adata_trans[:, all_genes].X.toarray() if hasattr(adata_trans.X, 'toarray') else adata_trans[:, all_genes].X
    
    gene_means = expr_matrix.mean(axis=0)
    gene_stds = expr_matrix.std(axis=0)
    gene_stds[gene_stds == 0] = 1
    expr_matrix = np.clip((expr_matrix - gene_means) / gene_stds, -2, 2)
    
    # 3. Plot Heatmap
    im = ax.imshow(expr_matrix.T, aspect='auto', cmap='RdBu_r',
                   vmin=-2, vmax=2, interpolation='nearest', rasterized=True)
    
    # 4. Plot Annotations
    trans_indices = range(*transition.indices(len(tcs_sorted)))
    trans_start, trans_stop = trans_indices.start, trans_indices.stop
    rect = patches.Rectangle((trans_start, -0.5), trans_stop - trans_start,
                             len(all_genes), linewidth=2.5, edgecolor='black',
                             facecolor='none', linestyle='--', zorder=10)
    ax.add_patch(rect)
    
    y_pos = 0
    category_positions = []
    for label in ['MS↓', 'IH↓', 'TD', 'IH↑', 'MS↑']:
        n_genes_cat = gene_labels.count(label)
        if n_genes_cat > 0:
            category_positions.append((y_pos + n_genes_cat/2 - 0.5, label))
            y_pos += n_genes_cat
            if y_pos < len(all_genes):
                ax.axhline(y_pos - 0.5, color='white', linewidth=2, zorder=5)
    
    # 5. Add Labels and Titles
    ax.set_yticks(range(len(all_genes)))
    ax.set_yticklabels(all_genes, fontsize=6)
    
    for center, label in category_positions:
        ax.text(len(tcs_sorted) * 1.02, center, label, # Position relative to axis
                ha='left', va='center', rotation=90, 
                fontsize=9, fontweight='bold', clip_on=False) 
    
    ax.set_ylabel('Genes', fontsize=9)
    xlabel = "Cells" if max_cells else "Metacells"
    ax.set_xlabel(f'{xlabel} ordered by TCS (A{source_attr} → A{target_attr})', fontsize=9)
    if title:
        ax.set_title(title, fontsize=10, fontweight='bold')
    
    # 6. Add Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.06) 
    cbar.set_label('Expression (z-score)', rotation=270, labelpad=12, fontsize=7)
    
    # 7. Add TCS inset plot
    ax_tcs = ax.inset_axes([0, -0.12, 1, 0.04])
    ax_tcs.plot(range(len(tcs_sorted)), tcs_sorted, color='green', linewidth=2)
    ax_tcs.axvspan(trans_start, trans_stop, alpha=0.2, color='gray')
    ax_tcs.set_xlim(0, len(tcs_sorted)); ax_tcs.set_ylim(0, 1)
    ax_tcs.set_xlabel('TCS', fontsize=7); ax_tcs.set_yticks([0, 1])
    ax_tcs.tick_params(labelsize=6)
    
    
def plot_transcendental_heatmap_cell(adata, source_attr, target_attr, ax,
                                n_genes_per_category=5, max_cells=None, 
                                region_method='logistic', title=None, fig_dir='.', cluster_name='leiden_0.5'):
    """
    Master function to create a transcendental analysis heatmap.
    
    This function calls src.transcendental helpers to get data and then focuses *only* on plotting.
    """
    adata.obs[cluster_name] = adata.obs[cluster_name].astype(str)
    mask = (adata.obs[cluster_name] == source_attr) | (adata.obs[cluster_name] == target_attr)
    
    if mask.sum() < 10:
        ax.text(0.5, 0.5, f'Insufficient data\nC{source_attr}→C{target_attr}',
               ha='center', va='center', transform=ax.transAxes); return

    adata_trans = adata[mask].copy()
    tcs = tran.calculate_tcs(adata_trans, source_attr, target_attr)
    sort_idx = np.argsort(tcs)[::-1] # Sort descending
    adata_trans = adata_trans[sort_idx].copy()
    tcs_sorted = tcs[sort_idx]
    
    if max_cells and len(tcs_sorted) > max_cells:
        sample_idx = np.linspace(0, len(tcs_sorted)-1, max_cells, dtype=int)
        adata_trans = adata_trans[sample_idx].copy()
        tcs_sorted = tcs_sorted[sample_idx]
    
    # 1. Get analysis from src.transcendental
    regions = tran.identify_transition_regions(tcs_sorted, method=region_method)
    stable_source, transition, stable_target = regions
    
    td_genes, td_genes_scores = tran.identify_td_genes(adata_trans, tcs_sorted, regions, n_genes=n_genes_per_category)
    
    columns = ['Genes', 'abs(corr)', 'corr', 'pval']
    pd_td_genes = pd.DataFrame(td_genes_scores, columns=columns)
    pd_td_genes['Category'] = 'TD'
    
    ms_down, ms_up, ih_down, ih_up = tran.identify_ms_ih_genes(
        adata_trans, regions, td_genes, n_genes=n_genes_per_category
    )
    
    dfs_to_combine = [pd_td_genes]
    additional_genes = {
                        'MS_Down': ms_down,
                        'MS_Up': ms_up,
                        'IH_Down': ih_down,
                        'IH_Up': ih_up
                    }
    
    for category, gene_list in additional_genes.items():
        # Create a temporary DF for each list
        temp_df = pd.DataFrame({'Genes': gene_list})
        temp_df['Category'] = category
        dfs_to_combine.append(temp_df)
        
    final_df = pd.concat(dfs_to_combine, ignore_index=True)
    
    cell_seacell = 'seacell' if 'Metacell' in title else 'cell'
    csv_file_name = fig_dir / f"td_ms_ih_genes_scores_{source_attr}_to_{target_attr}_{cell_seacell}_original_clusters.csv"
    cols = ['Category'] + [c for c in final_df.columns if c != 'Category']
    final_df = final_df[cols]
    final_df.to_csv(csv_file_name, index=False)
    print(f"\nAll Genes (TD, MS, IH) saved to {csv_file_name}.")
    
    all_genes = ms_down + ih_down + td_genes + ih_up + ms_up
    gene_labels = (['MS↓'] * len(ms_down) + ['IH↓'] * len(ih_down) +
                   ['TD'] * len(td_genes) + ['IH↑'] * len(ih_up) + ['MS↑'] * len(ms_up))
    
    if len(all_genes) == 0:
        ax.text(0.5, 0.5, 'No significant genes found', ha='center', va='center', transform=ax.transAxes); return

    # 2. Prepare expression matrix for plotting
    all_genes = [g for g in all_genes if g in adata_trans.var_names]
    expr_matrix = adata_trans[:, all_genes].X.toarray() if hasattr(adata_trans.X, 'toarray') else adata_trans[:, all_genes].X
    
    gene_means = expr_matrix.mean(axis=0)
    gene_stds = expr_matrix.std(axis=0)
    gene_stds[gene_stds == 0] = 1
    expr_matrix = np.clip((expr_matrix - gene_means) / gene_stds, -2, 2)
    
    # 3. Plot Heatmap
    im = ax.imshow(expr_matrix.T, aspect='auto', cmap='RdBu_r',
                   vmin=-2, vmax=2, interpolation='nearest', rasterized=True)
    
    # 4. Plot Annotations
    trans_indices = range(*transition.indices(len(tcs_sorted)))
    trans_start, trans_stop = trans_indices.start, trans_indices.stop
    rect = patches.Rectangle((trans_start, -0.5), trans_stop - trans_start,
                             len(all_genes), linewidth=2.5, edgecolor='black',
                             facecolor='none', linestyle='--', zorder=10)
    ax.add_patch(rect)
    
    y_pos = 0
    category_positions = []
    for label in ['MS↓', 'IH↓', 'TD', 'IH↑', 'MS↑']:
        n_genes_cat = gene_labels.count(label)
        if n_genes_cat > 0:
            category_positions.append((y_pos + n_genes_cat/2 - 0.5, label))
            y_pos += n_genes_cat
            if y_pos < len(all_genes):
                ax.axhline(y_pos - 0.5, color='white', linewidth=2, zorder=5)
    
    # 5. Add Labels and Titles
    ax.set_yticks(range(len(all_genes)))
    ax.set_yticklabels(all_genes, fontsize=16)
    
    for center, label in category_positions:
        ax.text(len(tcs_sorted) * 1.02, center, label, # Position relative to axis
                ha='left', va='center', rotation=90, 
                fontsize=16, fontweight='bold', clip_on=False) 
    
    ax.set_ylabel('Genes', fontsize=16)
    xlabel = "Cells" if max_cells else "Metacells"
    ax.set_xlabel(f'{xlabel} ordered by TCS (C{source_attr} → C{target_attr})', fontsize=12)
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold')
    
    # 6. Add Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.06) 
    cbar.set_label('Expression (z-score)', rotation=270, labelpad=12, fontsize=10)
    
    # 7. Add TCS inset plot
    ax_tcs = ax.inset_axes([0, -0.12, 1, 0.04])
    ax_tcs.plot(range(len(tcs_sorted)), tcs_sorted, color='green', linewidth=2)
    ax_tcs.axvspan(trans_start, trans_stop, alpha=0.2, color='gray')
    ax_tcs.set_xlim(0, len(tcs_sorted)); ax_tcs.set_ylim(0, 1)
    ax_tcs.set_xlabel('TCS', fontsize=10); ax_tcs.set_yticks([0, 1])
    ax_tcs.tick_params(labelsize=6)