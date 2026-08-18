import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

def calculate_tcs(adata_trans, source_attr, target_attr):
    """
    Calculate Transition Cell Score (TCS) - Equation (8) from Methods.
    This function robustly handles both single-cell (mapped) and metacell adatas.
    """
    if 'rho_class' in adata_trans.obsm:
        rho_class = adata_trans.obsm['rho_class']
    elif 'membership' in adata_trans.obsm and isinstance(adata_trans.obsm['membership'], np.ndarray):
        rho_class = adata_trans.obsm['membership']
    elif 'da_out' in adata_trans.uns and 'rho_class' in adata_trans.uns['da_out']:
        rho_class = adata_trans.uns['da_out']['rho_class']
    else:
        # Fallback for adatas where membership might not be present
        tcs = (adata_trans.obs['attractor'].astype(str) == target_attr).astype(float)
        return tcs

    try:
        source_idx = int(source_attr)
        target_idx = int(target_attr)
        rho_i = rho_class[:, source_idx]
        rho_j = rho_class[:, target_idx]
        tcs = rho_i / (rho_i + rho_j + 1e-10)
        return tcs
    except (ValueError, IndexError):
        tcs = (adata_trans.obs['attractor'].astype(str) == target_attr).astype(float)
        return tcs

def map_seacell_memberships_to_cells(adata_org, adata_seacell):
    """Map SEACell membership functions to individual cells."""
    if 'membership' in adata_seacell.obsm:
        rho_class_seacell = adata_seacell.obsm['membership']
    elif 'da_out' in adata_seacell.uns and 'rho_class' in adata_seacell.uns['da_out']:
        rho_class_seacell = adata_seacell.uns['da_out']['rho_class']
    else:
        raise ValueError("rho_class not found in expected locations in adata_seacell.")

    n_seacells, n_attractors = rho_class_seacell.shape
    seacell_names = adata_seacell.obs_names
    seacell_to_idx = {name: idx for idx, name in enumerate(seacell_names)}
    rho_class_cells = np.zeros((adata_org.n_obs, n_attractors))
    
    n_mapped = 0
    for i, seacell_name in enumerate(adata_org.obs['SEACell']):
        if pd.notna(seacell_name) and str(seacell_name) in seacell_to_idx:
            seacell_idx = seacell_to_idx[str(seacell_name)]
            rho_class_cells[i, :] = rho_class_seacell[seacell_idx, :]
            n_mapped += 1
            
    adata_org.obsm['rho_class'] = rho_class_cells
    if 'da_out' not in adata_org.uns:
        adata_org.uns['da_out'] = {}
    adata_org.uns['da_out']['rho_class'] = rho_class_cells
    print(f"Successfully mapped {n_mapped} cells to SEACell memberships.")
    return adata_org

def identify_transition_regions(tcs_sorted, method='logistic'):
    """Identify stable and transition regions from sorted TCS scores."""
    n_cells = len(tcs_sorted)
    
    if method == 'logistic':
        def logistic(x, L, x0, k):
            return L / (1 + np.exp(-k * (x - x0)))
        x_data = np.arange(n_cells)
        try:
            popt, _ = curve_fit(logistic, x_data, tcs_sorted, p0=[1, n_cells/2, 0.01], maxfev=5000)
            L, x0, k = popt
            center = int(x0)
            width = int(2.2 / abs(k)) # Width of the transition
            trans_start = max(0, center - width//2)
            trans_end = min(n_cells, center + width//2)
        except:
            method = 'adaptive' # Fallback
            
    if method == 'adaptive':
        trans_mask = (tcs_sorted > 0.2) & (tcs_sorted < 0.8)
        trans_indices = np.where(trans_mask)[0]
        if len(trans_indices) > 0:
            trans_start = trans_indices[0]
            trans_end = trans_indices[-1] + 1
        else: # Fallback if no cells in 0.2-0.8 range
            trans_start = n_cells // 3
            trans_end = 2 * n_cells // 3
            
    return slice(0, trans_start), slice(trans_start, trans_end), slice(trans_end, n_cells)

def identify_td_genes(adata_trans, tcs_sorted, regions, corr_threshold=0.3, n_genes=5):
    """Identify Transition-Driver (TD) genes."""
    _, transition, _ = regions
    tcs_trans = tcs_sorted[transition]
    if len(tcs_trans) < 1: return []
    
    X = adata_trans.X.toarray() if hasattr(adata_trans.X, 'toarray') else adata_trans.X
    X_trans = X[transition, :]
    td_genes = []
    
    for i, gene in enumerate(adata_trans.var_names):
        expr_trans = X_trans[:, i]
        if np.std(expr_trans) == 0: continue
        try:
            corr, pval = stats.pearsonr(tcs_trans, expr_trans)
            if abs(corr) > corr_threshold and pval < 0.05:
                td_genes.append((gene, abs(corr), corr, pval))
        except: continue
            
    td_genes.sort(key=lambda x: x[1], reverse=True)
    
    return [g[0] for g in td_genes[:n_genes]], td_genes

def identify_ms_ih_genes(adata_trans, regions, td_genes, n_genes=5, pval_threshold=0.01):
    """Identify Meta-Stable (MS) and Intermediate-Hybrid (IH) genes."""
    stable_source, transition, stable_target = regions
    X = adata_trans.X.toarray() if hasattr(adata_trans.X, 'toarray') else adata_trans.X
    
    ms_genes_down, ms_genes_up, ih_genes_down, ih_genes_up = [], [], [], []
    excluded_genes = set(td_genes)
    
    def get_mean(expr_slice):
        return np.mean(expr_slice) if len(expr_slice) > 0 else 0
        
    def get_ttest(slice1, slice2):
        if len(slice1) < 3 or len(slice2) < 3:
            return 1.0, 0.0
        try:
            _, pval = stats.ttest_ind(slice1, slice2)
            mean1, mean2 = np.mean(slice1), np.mean(slice2)
            return pval, mean1 - mean2
        except:
            return 1.0, 0.0

    for i, gene in enumerate(adata_trans.var_names):
        if gene in excluded_genes: continue
        
        expr = X[:, i]
        expr_s, expr_t, expr_target = expr[stable_source], expr[transition], expr[stable_target]
        
        pval_s_t, diff_s_t = get_ttest(expr_s, expr_t)
        if pval_s_t < pval_threshold and diff_s_t > 0: # Down in transition
            ms_genes_down.append((gene, diff_s_t))
            
        pval_target_t, diff_target_t = get_ttest(expr_target, expr_t)
        if pval_target_t < pval_threshold and diff_target_t > 0: # Up from transition
            ms_genes_up.append((gene, diff_target_t))
            
        pval_s_target, diff_s_target = get_ttest(expr_s, expr_target)
        if pval_s_target < pval_threshold:
            is_ms = gene in [g[0] for g in ms_genes_down] or gene in [g[0] for g in ms_genes_up]
            if not is_ms:
                if diff_s_target > 0: # Down-regulated (IH-down)
                    ih_genes_down.append((gene, diff_s_target))
                else: # Up-regulated (IH-up)
                    ih_genes_up.append((gene, abs(diff_s_target)))

    ms_genes_down.sort(key=lambda x: x[1], reverse=True)
    ms_genes_up.sort(key=lambda x: x[1], reverse=True)
    ih_genes_down.sort(key=lambda x: x[1], reverse=True)
    ih_genes_up.sort(key=lambda x: x[1], reverse=True)
    
    return ([g[0] for g in ms_genes_down[:n_genes]],
            [g[0] for g in ms_genes_up[:n_genes]],
            [g[0] for g in ih_genes_down[:n_genes]],
            [g[0] for g in ih_genes_up[:n_genes]])