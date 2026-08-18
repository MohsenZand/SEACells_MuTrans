#!/usr/bin/env python3
"""
Supplementary pipeline: SEACells + MuTrans on the LARRY lineage-tracing dataset
(GSE140802, NM trajectory).

Runs the full process end-to-end on a single input file:
  1. SEACells:  92.5k cells -> ~1.2k metacells (kernel built on the existing X_pca)
  2. MuTrans:   dynamical analysis with K_cluster = 14 attractors

Notes specific to this dataset (differ from the iPSC pipeline):
  - Only total-count-normalized counts are available (adata.X). There is no raw
    layer and no .raw, so metacells are summarized directly from X.
  - X_pca (50 dims) is already present, so SEACells reuses it instead of
    recomputing PCA.

Each stage is guarded by an os.path.exists check on its output, so re-running
resumes from the last completed stage. Delete an output to force recomputation.

Run on a compute node:
    python LARRY/scripts/supp_larry.py
"""
import os
import sys
import random
import collections.abc

import numpy as np
import pandas as pd
import scanpy as sc
import SEACells
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# --- Make 'from src import ...' resolve, and import shared parameters ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import config  # reuse tuned numeric parameters only (paths overridden below)

# --- MuTrans (not on PyPI; install and point MUTRANS_PATH at the checkout) ---
sys.path.append(config.MUTRANS_PATH)
from MuTrans.Example import pyMuTrans as pm

np.random.seed(0)
random.seed(0)
sns.set_style('ticks')
matplotlib.rcParams['figure.figsize'] = [4, 4]
matplotlib.rcParams['figure.dpi'] = 100

# ---------------------------------------------------------------------------
# Dataset-specific configuration
# ---------------------------------------------------------------------------
# Download GSE140802 (LARRY) from GEO and set LARRY_RAW_H5AD, or place it under data/raw/.
LARRY_RAW = os.environ.get(
    'LARRY_RAW_H5AD',
    str(config.RAW_DATA_DIR / 'larry_NM_trajectory_normedCounts.h5ad'))

TARGET_METACELLS = 1200          # 92.5k / ~77 cells per metacell
BUILD_KERNEL_ON = 'X_pca'        # precomputed 50-dim PCA already in the file
CELLTYPE_LABEL = 'Cell.type.clean'   # 11 annotated cell types (for purity)

# Output layout under the experiments project (data/ and results/ are gitignored)
OUT_SEACELL_DIR = config.PROCESSED_DATA_DIR / 'larry' / 'seacells'
OUT_MUTRANS_DIR = config.PROCESSED_DATA_DIR / 'larry' / 'mutrans'
RESULTS_DIR = config.RESULTS_DIR / 'larry'
FIG_DIR = RESULTS_DIR / 'figures'
for d in [OUT_SEACELL_DIR, OUT_MUTRANS_DIR, RESULTS_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)
sc.settings.figdir = FIG_DIR

SEACELL_AD_ORG = OUT_SEACELL_DIR / 'larry_with_SEACells.h5ad'
SEACELL_AD_SUMMARY = OUT_SEACELL_DIR / 'larry_SEACell_summary.h5ad'
MUTRANS_AD_SEACELL = OUT_MUTRANS_DIR / 'larry_seacells_mutrans.h5ad'
MUTRANS_AD_ORG = OUT_MUTRANS_DIR / 'larry_seacells_org_mutrans.h5ad'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_and_convert_matlab_objects(obj, key_path=None):
    """Recursively replace MATLAB-typed values in a dict (left by MuTrans in
    .uns) with numpy arrays, so the AnnData can be written to h5ad."""
    if key_path is None:
        key_path = []
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
            print(f"  Converting '{k}' at {' -> '.join(key_path + [k])} to numpy array.")
            try:
                obj[k] = np.asarray(v)
            except Exception:
                obj[k] = str(v)


def _save(fig_name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / fig_name, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Stage 1: SEACells
# ---------------------------------------------------------------------------
def run_seacells_stage():
    if SEACELL_AD_SUMMARY.exists() and SEACELL_AD_ORG.exists():
        print(f"[SEACells] Outputs already exist -- skipping.\n"
              f"  {SEACELL_AD_ORG}\n  {SEACELL_AD_SUMMARY}")
        return

    print("\n" + "=" * 80)
    print("Stage 1: SEACells computation")
    print("=" * 80)

    print(f"Loading {LARRY_RAW}")
    ad = sc.read(LARRY_RAW)
    print(f"  shape: {ad.shape}")
    ad.X = ad.X.astype(np.float32)

    if BUILD_KERNEL_ON not in ad.obsm:
        raise KeyError(f"{BUILD_KERNEL_ON} not found in adata.obsm; available: {list(ad.obsm)}")

    n_SEACells = TARGET_METACELLS
    print(f"Running SEACells: n_SEACells = {n_SEACells}, kernel on {BUILD_KERNEL_ON}")
    model = SEACells.core.SEACells(
        ad,
        build_kernel_on=BUILD_KERNEL_ON,
        n_SEACells=n_SEACells,
        n_waypoint_eigs=config.SEACELL_N_WAYPOINT_EIGS,
        convergence_epsilon=config.SEACELL_CONVERGENCE_EPS,
    )
    model.construct_kernel_matrix()
    model.initialize_archetypes()
    print("Fitting model...")
    model.fit(min_iter=10, max_iter=50)
    try:
        model.plot_convergence(save_as=str(FIG_DIR / "rss_convergence.png"))
    except Exception as e:
        print(f"  (convergence plot skipped: {e})")

    ad.obs['SEACell'] = model.get_hard_assignments()
    print(f"Saving ORG -> {SEACELL_AD_ORG.name}")
    ad.write(SEACELL_AD_ORG)

    # Quality evaluation (best-effort)
    try:
        purity = SEACells.evaluate.compute_celltype_purity(ad, CELLTYPE_LABEL)
        plt.figure(figsize=(4, 4)); sns.boxplot(data=purity, y=f'{CELLTYPE_LABEL}_purity')
        plt.title('Celltype purity'); sns.despine(); _save("celltype_purity.png")
        compactness = SEACells.evaluate.compactness(ad, 'X_pca')
        plt.figure(figsize=(4, 4)); sns.boxplot(data=compactness, y='compactness')
        plt.title('Compactness'); sns.despine(); _save("compactness.png")
        separation = SEACells.evaluate.separation(ad, 'X_pca', nth_nbr=1)
        plt.figure(figsize=(4, 4)); sns.boxplot(data=separation, y='separation')
        plt.title('Separation'); sns.despine(); _save("separation.png")
    except Exception as e:
        print(f"  (evaluation plots skipped: {e})")

    # Aggregate to metacells from the normalized X (no raw layer exists).
    # summarize_layer='X' sums adata.X and stores the result in both .X and
    # .layers['raw'], which is what the MuTrans stage reads.
    print("Summarizing metacells from normalized X...")
    SEACell_ad = SEACells.core.summarize_by_SEACell(
        ad, SEACells_label='SEACell', celltype_label=CELLTYPE_LABEL, summarize_layer='X'
    )
    print(f"  metacell summary shape: {SEACell_ad.shape}")
    print(f"Saving SUMMARY -> {SEACELL_AD_SUMMARY.name}")
    SEACell_ad.write(SEACELL_AD_SUMMARY)
    print("Stage 1 complete.")


# ---------------------------------------------------------------------------
# Stage 2: MuTrans
# ---------------------------------------------------------------------------
def preprocess_and_cluster(adata_seacell):
    print("\n[MuTrans 1/4] Preprocessing and clustering metacells...")
    print(f"  shape before filtering: {adata_seacell.shape}")
    sc.pp.filter_cells(adata_seacell, min_genes=300)
    sc.pp.filter_genes(adata_seacell, min_cells=10)
    print(f"  after filtering: {adata_seacell.shape}")

    # Mouse gene categories
    adata_seacell.var["mt"] = adata_seacell.var_names.str.startswith("mt-")
    adata_seacell.var["ribo"] = adata_seacell.var_names.str.startswith(("Rps", "Rpl"))
    adata_seacell.var["hb"] = adata_seacell.var_names.str.contains(r"^Hb[ab]")
    sc.pp.calculate_qc_metrics(adata_seacell, qc_vars=["mt", "ribo", "hb"],
                               inplace=True, log1p=True)
    try:
        sc.pl.violin(adata_seacell,
                     ["n_genes_by_counts", "total_counts", "pct_counts_mt",
                      "pct_counts_ribo", "pct_counts_hb"],
                     jitter=0.4, multi_panel=True, save='_qc.pdf', show=False)
    except Exception as e:
        print(f"  (QC violin skipped: {e})")

    # Holly suggested to rerun (after SEACell) without pp.normalize_total
    # sc.pp.normalize_total(adata_seacell, target_sum=1e4) # this line was done in v1  
    sc.pp.log1p(adata_seacell)
    sc.pp.highly_variable_genes(adata_seacell, n_top_genes=config.N_TOP_HVG)
    adata_seacell = adata_seacell[:, adata_seacell.var.highly_variable]
    print(f"  after HVG selection: {adata_seacell.shape}")

    sc.pp.pca(adata_seacell)
    sc.pp.neighbors(adata_seacell)
    sc.tl.umap(adata_seacell)
    # Another suggestion from Holly to use leiden_0.6 when we rerun
    sc.tl.leiden(adata_seacell, resolution=config.LEIDEN_RESOLUTION, key_added='leiden_0.5')
    try:
        sc.pl.umap(adata_seacell, color=['leiden_0.5'], legend_loc='on data',
                   save='_leiden_0.5.pdf', show=False)
    except Exception as e:
        print(f"  (UMAP plot skipped: {e})")
    return adata_seacell


def run_mutrans(adata_seacell):
    print("\n[MuTrans 2/4] Running dynamical analysis (K_cluster = "
          f"{config.MUTRAMS_K_CLUSTER})...")
    par = {
        "choice_distance": "cosine",
        "perplex": config.MUTRAMS_PERPLEX,
        "K_cluster": config.MUTRAMS_K_CLUSTER,   # 14 attractors
        "reduction_coord": 'umap',
        "weight_scale": False,
        "write_anndata": True,
        "learn_regulatory_network": True,
        "generate_interactive_vis": True,
        "reduce_large_scale": False,
        "force_double_precision": True,
    }
    return pm.dynamical_analysis(adata_seacell, par)


def transfer_and_save(adata_seacell, adata_org):
    print("\n[MuTrans 3/4] Transferring metrics to single cells and saving...")
    ent_map = adata_seacell.obs['entropy'].to_dict()
    adata_org.obs['entropy'] = adata_org.obs['SEACell'].map(ent_map)
    for key in ['land', 'attractor']:
        if key in adata_seacell.obs:
            m = adata_seacell.obs[key].to_dict()
            adata_org.obs[key] = adata_org.obs['SEACell'].map(m)

    find_and_convert_matlab_objects(adata_org.uns)
    adata_org.write(MUTRANS_AD_ORG)
    print(f"  Saved: {MUTRANS_AD_ORG.name}")

    find_and_convert_matlab_objects(adata_seacell.uns)
    adata_seacell.write(MUTRANS_AD_SEACELL)
    print(f"  Saved: {MUTRANS_AD_SEACELL.name}")
    return adata_seacell, adata_org


def lineage_network(adata_seacell):
    print("\n[MuTrans 4/4] Lineage inference and network extraction...")
    try:
        plt.figure(figsize=(10, 6))
        pm.infer_lineage(adata_seacell, method="MPFT", flux_fraction=0.3,
                         size_point=40, alpha_point=0.5, size_text=20, color_palette=None)
        plt.savefig(FIG_DIR / 'mpft_transition.pdf', bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"  (MPFT plot skipped: {e})")

    transition_matrix = adata_seacell.uns['land']['land']
    N_land = transition_matrix.shape[0]
    land_ids = adata_seacell.obs_names
    if len(land_ids) != N_land:
        land_ids = land_ids[:N_land]

    SCORE_THRESHOLD = 1e-6
    rows = []
    for i in range(N_land):
        for j in range(N_land):
            score = transition_matrix[i, j]
            if score > SCORE_THRESHOLD:
                rows.append((land_ids[i], land_ids[j], score))
    network_df = pd.DataFrame(rows, columns=['Source_LAND', 'Target_LAND', 'Linkage_Score'])
    network_df.to_csv(RESULTS_DIR / 'land_lineage_network_table.csv', index=False)
    print(f"  Extracted {len(network_df)} linkages from {N_land}x{N_land} matrix")

    attr_map = adata_seacell.obs['attractor'].astype(str)
    network_df['Source_Attractor'] = network_df['Source_LAND'].map(attr_map)
    network_df['Target_Attractor'] = network_df['Target_LAND'].map(attr_map)
    agg = (network_df.groupby(['Source_Attractor', 'Target_Attractor'])['Linkage_Score']
           .sum().reset_index()
           .rename(columns={'Source_Attractor': 'Source_Attractor_ID',
                            'Target_Attractor': 'Target_Attractor_ID',
                            'Linkage_Score': 'Total_Linkage_Score'}))
    agg.to_csv(RESULTS_DIR / 'attractor_lineage_network_aggregated.csv', index=False)
    print(f"  Aggregated to {len(agg)} attractor-to-attractor linkages")


def run_mutrans_stage():
    if MUTRANS_AD_SEACELL.exists() and MUTRANS_AD_ORG.exists():
        print(f"[MuTrans] Outputs already exist -- skipping.\n"
              f"  {MUTRANS_AD_ORG}\n  {MUTRANS_AD_SEACELL}")
        return

    print("\n" + "=" * 80)
    print("Stage 2: MuTrans analysis")
    print("=" * 80)
    adata_org = sc.read(SEACELL_AD_ORG)
    adata_seacell = sc.read(SEACELL_AD_SUMMARY)
    print(f"  ORG: {adata_org.shape}   SUMMARY: {adata_seacell.shape}")

    adata_seacell = preprocess_and_cluster(adata_seacell)
    adata_seacell = run_mutrans(adata_seacell)
    adata_seacell, adata_org = transfer_and_save(adata_seacell, adata_org)
    lineage_network(adata_seacell)
    print("Stage 2 complete.")


def main():
    print("=" * 80)
    print("LARRY supplementary pipeline (SEACells -> MuTrans)")
    print("=" * 80)
    run_seacells_stage()
    run_mutrans_stage()
    print("\n" + "=" * 80)
    print("Pipeline complete.")
    print(f"  SEACells:  {SEACELL_AD_ORG}")
    print(f"             {SEACELL_AD_SUMMARY}")
    print(f"  MuTrans:   {MUTRANS_AD_ORG}")
    print(f"             {MUTRANS_AD_SEACELL}")
    print(f"  Figures:   {FIG_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
