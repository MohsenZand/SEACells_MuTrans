#!/usr/bin/env python3
"""
Cell-level supplementary figures for the LARRY SEACells+MuTrans results.

Reads ONLY obs + obsm['X_umap'] from the (large) single-cell MuTrans h5ad via
h5py, so the 4.5 GB expression matrix is never loaded -- safe to run on a
login node.

Produces three figures in results/larry/figures/:
  1. celllevel_attractor_vs_leiden_r0_6_anno_heatmap  -- correspondence heatmap
     between MuTrans attractor and the leiden_r0_6_anno annotation (cf. panel B:
     establishes attractor identity).
  2. celllevel_umap_anno_entropy_land                 -- 3-panel cell UMAP colored
     by leiden_r0_6_anno, entropy, land score (cf. panel C, + annotation).
  3. celllevel_landscore_histogram                    -- histogram of the cell-level
     land score (cf. panel D, without the data-driven cutoff line).
"""
import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import scanpy as sc
sc.settings.verbosity = 0
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 8


_PROJECT_ROOT = Path(__file__).resolve().parents[2]   # repo root
MUTRANS_DIR = str(_PROJECT_ROOT / 'data' / 'processed' / 'larry' / 'mutrans')
ORG_H5AD = f'{MUTRANS_DIR}/larry_seacells_org_mutrans.h5ad'
SEACELL_H5AD = f'{MUTRANS_DIR}/larry_seacells_mutrans.h5ad'
FIG_DIR = str(_PROJECT_ROOT / 'results' / 'larry' / 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Attractor transitions of interest (source -> target)
TRANSITIONS = [('8', '7'), ('8', '5'), ('1', '7'), ('1', '5'), ('1', '8')]


def read_categorical(obs, key, as_float=False):
    """Decode an AnnData categorical obs column stored as a group."""
    g = obs[key]
    if isinstance(g, h5py.Group):
        cats = g['categories'][:]
        codes = g['codes'][:]
        if as_float:
            cats = np.asarray(cats, dtype=float)
            out = np.full(codes.shape, np.nan)
            valid = codes >= 0
            out[valid] = cats[codes[valid]]
            return out
        cats = np.array([c.decode() if isinstance(c, bytes) else c for c in cats])
        out = np.array([cats[i] if i >= 0 else 'NA' for i in codes])
        return out
    arr = g[:]
    return arr.astype(float) if as_float else arr


def load():
    with h5py.File(ORG_H5AD, 'r') as f:
        obs = f['obs']
        df = pd.DataFrame({
            'anno': read_categorical(obs, 'leiden_r0_6_anno'),
            'attractor': read_categorical(obs, 'attractor'),
            'entropy': read_categorical(obs, 'entropy', as_float=True),
            'land': read_categorical(obs, 'land', as_float=True),
        })
        umap = f['obsm']['X_umap'][:]
    df['umap1'] = umap[:, 0]
    df['umap2'] = umap[:, 1]
    return df


def _attr_order(vals):
    u = pd.unique(vals)
    try:
        return sorted(u, key=lambda x: int(x))
    except (ValueError, TypeError):
        return sorted(u)


def _corr_tables(df):
    """Return (counts, row_norm, col_norm) crosstabs of attractor x anno with a
    diagonalized column order shared by the heatmap and dotplot."""
    attr_order = _attr_order(df['attractor'])
    ct = pd.crosstab(df['attractor'], df['anno']).reindex(index=attr_order)
    col_order = ct.div(ct.sum(0), axis=1).idxmax().sort_values(
        key=lambda s: [attr_order.index(v) for v in s]).index.tolist()
    ct = ct[col_order]
    return ct, ct.div(ct.sum(1), axis=0), ct.div(ct.sum(0), axis=1)


# ---------------------------------------------------------------------------
# Figure 1: attractor vs leiden_r0_6_anno correspondence heatmap
# ---------------------------------------------------------------------------
def fig1_heatmap(df):
    attr_order = _attr_order(df['attractor'])
    ct = pd.crosstab(df['attractor'], df['anno'])
    ct = ct.reindex(index=attr_order)
    # order annotation columns by the attractor where each is most frequent
    col_order = ct.div(ct.sum(0), axis=1).idxmax().sort_values(
        key=lambda s: [attr_order.index(v) for v in s]).index.tolist()
    ct = ct[col_order]
    prop = ct.div(ct.sum(1), axis=0)  # row-normalized: composition of each attractor

    fig, ax = plt.subplots(figsize=(9, 6.5))
    sns.heatmap(prop, cmap='Reds', vmin=0, vmax=1, linewidths=0.4, linecolor='white',
                cbar_kws={'label': 'Fraction of attractor cells', 'shrink': 0.6},
                annot=True, fmt='.2f', annot_kws={'size': 6}, ax=ax)
    ax.set_xlabel('leiden_r0_6_anno', fontsize=11)
    ax.set_ylabel('MuTrans attractor', fontsize=11)
    ax.set_title('Attractor identity vs leiden_r0_6_anno (cell level)',
                 fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    plt.setp(ax.get_xticklabels(), ha='right')
    ax.tick_params(axis='y', rotation=0, labelsize=9)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{FIG_DIR}/celllevel_attractor_vs_leiden_r0_6_anno_heatmap.{ext}',
                    dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  wrote fig1 (heatmap)')


# ---------------------------------------------------------------------------
# Figure 2: 3-panel cell-level UMAP
# ---------------------------------------------------------------------------
def _umap_continuous(ax, df, values, title, cbar_label):
    order = np.argsort(values)  # low values drawn first
    sc = ax.scatter(df['umap1'].values[order], df['umap2'].values[order],
                    c=values[order], cmap='viridis', s=1.2, alpha=0.8,
                    rasterized=True, edgecolors='none')
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=12, fontweight='bold', loc='left')
    ax.set_xlabel('UMAP1', fontsize=9); ax.set_ylabel('UMAP2', fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def fig2_umaps(df):
    fig, axs = plt.subplots(1, 3, figsize=(21, 6.5))

    # Panel a: categorical annotation
    ax = axs[0]
    cats = sorted(df['anno'].unique())
    palette = sns.color_palette('tab20', len(cats))
    cmap = {c: palette[i] for i, c in enumerate(cats)}
    order = np.random.RandomState(0).permutation(len(df))  # avoid overplot bias
    colors = df['anno'].map(cmap).values
    ax.scatter(df['umap1'].values[order], df['umap2'].values[order],
               c=colors[order], s=1.2, alpha=0.8, rasterized=True, edgecolors='none')
    handles = [plt.Line2D([0], [0], marker='o', linestyle='', markersize=5,
                          markerfacecolor=cmap[c], markeredgecolor='none', label=c)
               for c in cats]
    ax.legend(handles=handles, fontsize=6.5, loc='center left',
              bbox_to_anchor=(1.0, 0.5), frameon=False, title='leiden_r0_6_anno',
              title_fontsize=7)
    ax.set_title('leiden_r0_6_anno', fontsize=12, fontweight='bold', loc='left')
    ax.set_xlabel('UMAP1', fontsize=9); ax.set_ylabel('UMAP2', fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    _umap_continuous(axs[1], df, df['entropy'].values, 'entropy', 'entropy')
    _umap_continuous(axs[2], df, df['land'].values, 'land score', 'land score')

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{FIG_DIR}/celllevel_umap_anno_entropy_land.{ext}',
                    dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  wrote fig2 (umaps)')


# ---------------------------------------------------------------------------
# Figure 3: land score histogram (no cutoff line)
# ---------------------------------------------------------------------------
def fig3_hist(df):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    land = df['land'].values
    land = land[np.isfinite(land)]
    ax.hist(land, bins=80, color='0.5', edgecolor='none')
    ax.set_xlabel('metacell land score on cell level', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Cell-level land score distribution', fontsize=12, fontweight='bold')
    sns.despine(ax=ax)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{FIG_DIR}/celllevel_landscore_histogram.{ext}',
                    dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  wrote fig3 (histogram)')


def fig4_dotplot(df):
    """Panel-B-style dotplot of attractor identity via leiden_r0_6_anno.
    dot size  = fraction of the ATTRACTOR's cells in that annotation (row norm)
    dot color = fraction of the ANNOTATION's cells in that attractor (col norm)
    """
    ct, rownorm, colnorm = _corr_tables(df)
    attrs = list(rownorm.index)          # rows (y), attractor 0 at top
    annos = list(rownorm.columns)        # cols (x)
    ny, nx = len(attrs), len(annos)

    xs, ys, sizes, colors = [], [], [], []
    for yi, a in enumerate(attrs):
        for xi, c in enumerate(annos):
            frac = rownorm.loc[a, c]
            if frac <= 0:
                continue
            xs.append(xi)
            ys.append(ny - 1 - yi)       # put attractor 0 at the top
            sizes.append(frac * 420 + 8)
            colors.append(colnorm.loc[a, c])

    fig, ax = plt.subplots(figsize=(11, 6.8))
    scat = ax.scatter(xs, ys, s=sizes, c=colors, cmap='Reds', vmin=0, vmax=1,
                      edgecolors='0.35', linewidths=0.4, zorder=3)
    ax.set_xticks(range(nx)); ax.set_xticklabels(annos, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(ny)); ax.set_yticklabels(attrs[::-1], fontsize=9)
    ax.set_xlim(-0.6, nx - 0.4); ax.set_ylim(-0.6, ny - 0.4)
    ax.set_xlabel('leiden_r0_6_anno', fontsize=11)
    ax.set_ylabel('MuTrans attractor', fontsize=11)
    ax.set_title('Metacell attractor identity via leiden_r0_6_anno (cell level)',
                 fontsize=12, fontweight='bold')

    cbar = fig.colorbar(scat, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label('Fraction of annotation in attractor', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # size legend (pushed clear of the colorbar)
    for frac in (0.25, 0.5, 0.75, 1.0):
        ax.scatter([], [], s=frac * 420 + 8, c='0.6', edgecolors='0.35',
                   linewidths=0.4, label=f'{int(frac*100)}%')
    ax.legend(title='Fraction of\nattractor in group', loc='center left',
              bbox_to_anchor=(1.30, 0.5), frameon=False, labelspacing=1.4,
              fontsize=7, title_fontsize=7, borderpad=1)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{FIG_DIR}/celllevel_attractor_vs_leiden_r0_6_anno_dotplot.{ext}',
                    dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  wrote fig4 (dotplot)')


# ---------------------------------------------------------------------------
# Transition analysis (HEAVY: needs pyMuTrans + the 4.5 GB cell file).
# Reuses src.plotting / src.transcendental exactly as scripts 02/03/04 do, so
# per transition it emits, in results/larry/figures/:
#   - transition_{s}to{t}.{png,pdf}                      MPPT lineage plot (panel-1 style)
#   - larry_transition_{s}_to_{t}_C_D_heatmaps.{pdf,png} reduced comprehensive figure:
#         panel C (violin entropy), panel D (violin land), + the two bottom
#         transcendental heatmaps (cell-level and metacell-level)
#   - td_genes_scores_{s}_to_{t}_cell.csv                TD-gene scores (cell level)
#   - td_genes_scores_{s}_to_{t}_seacell.csv             TD-gene scores (metacell level)
# The combined figure PDF is the "pdf version" of the two gene-score tables.
# ---------------------------------------------------------------------------
def _safe_heatmap(pl, adata, si, sf, ax, fig_dir, title, preferred, max_cells):
    """Draw a transcendental heatmap, falling back to 'adaptive' region
    detection if the preferred method yields an empty transition region
    (identify_td_genes returns [] there, which crashes the unpack in
    src.plotting -- a bug we route around without modifying src)."""
    methods = [preferred] + (['adaptive'] if preferred != 'adaptive' else [])
    for method in methods:
        try:
            ax.clear()
            pl.plot_transcendental_heatmap(
                adata, si, sf, ax, max_cells=max_cells, region_method=method,
                title=title, fig_dir=fig_dir)
            if method != preferred:
                print(f'    ({title}: fell back to region_method=adaptive)')
            return
        except Exception as e:
            print(f'    heatmap "{title}" method={method} failed: {e}')
    ax.clear()
    ax.text(0.5, 0.5, f'heatmap failed\n{title}', ha='center', va='center',
            transform=ax.transAxes)
    ax.axis('off')


def generate_transition_figures():
    proj = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if proj not in sys.path:
        sys.path.append(proj)
    from src import config
    from src import plotting as pl          # imports pyMuTrans at module load
    from src import transcendental as tran
    sys.path.append(config.MUTRANS_PATH)
    from MuTrans.Example import pyMuTrans as pm

    # --- Robust transition-region detection (monkey-patch, does not touch src) ---
    # These attractor pairs are sharply bimodal (≈no cells with intermediate TCS).
    # src.identify_transition_regions('adaptive') then collapses the transition to a
    # single cell when exactly one intermediate exists (-> constant TCS -> 0 genes).
    # Replace only such degenerate regions with a band centred on the TCS=0.5 crossing.
    _orig_regions = tran.identify_transition_regions

    def _robust_regions(tcs_sorted, method='logistic'):
        ss, tr, st = _orig_regions(tcs_sorted, method=method)
        n = len(tcs_sorted)
        if (tr.stop - tr.start) >= max(10, int(0.05 * n)):
            return ss, tr, st                       # keep genuine transition windows
        t = np.asarray(tcs_sorted)
        below = np.where(t < 0.5)[0]                # sorted desc: first TCS<0.5 = boundary
        center = int(below[0]) if below.size else n // 2
        half = max(n // 6, 5)
        a, b = max(0, center - half), min(n, center + half)
        if b <= a:
            a, b = max(0, n // 3), min(n, 2 * n // 3)
        return slice(0, a), slice(a, b), slice(b, n)

    tran.identify_transition_regions = _robust_regions

    fig_dir = Path(FIG_DIR)

    print('Loading MuTrans objects (metacell + 4.5 GB cell)...')
    adata_seacell = sc.read(SEACELL_H5AD)
    adata_org = sc.read(ORG_H5AD)
    adata_seacell.obs['attractor'] = adata_seacell.obs['attractor'].astype(str)
    adata_org.obs['attractor'] = adata_org.obs['attractor'].astype(str)
    # entropy is stored categorical (mapped from metacells) -> make numeric for violins
    for k in ('entropy', 'land'):
        adata_org.obs[k] = pd.to_numeric(adata_org.obs[k].astype(str), errors='coerce')

    # Map metacell soft memberships onto single cells once (needed for cell-level TCS)
    if 'rho_class' not in adata_org.obsm:
        adata_org = tran.map_seacell_memberships_to_cells(adata_org, adata_seacell)

    for si, sf in TRANSITIONS:
        print(f'\n=== Transition {si} -> {sf} ===')

        # (1) MPPT lineage plot  ->  transition_{si}to{sf}.png/.pdf
        try:
            plt.figure(figsize=(8, 6))
            pm.infer_lineage(adata_seacell, si=int(si), sf=int(sf), method='MPPT',
                             flux_fraction=0.3, size_point=40, alpha_point=0.5)
            for ext in ('png', 'pdf'):
                plt.savefig(f'{FIG_DIR}/transition_{si}to{sf}.{ext}',
                            dpi=150, bbox_inches='tight')
            plt.close()
            print(f'  wrote transition_{si}to{sf}')
        except Exception as e:
            plt.close()
            print(f'  MPPT {si}->{sf} FAILED: {e}')

        # (2+3) reduced comprehensive figure: C, D violins + cell & metacell heatmaps.
        # The two heatmap calls also write the td_genes_scores_*_{cell,seacell}.csv files.
        fig = plt.figure(figsize=(24, 13))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.28,
                               height_ratios=[1, 1.5])
        ax_c = fig.add_subplot(gs[0, 0])
        pl.plot_violin(adata_org, ax_c, 'entropy', 'C', groupby='attractor')
        ax_d = fig.add_subplot(gs[0, 1])
        pl.plot_violin(adata_org, ax_d, 'land', 'D', groupby='attractor')

        ax_i = fig.add_subplot(gs[1, 0])
        _safe_heatmap(pl, adata_org, si, sf, ax_i, fig_dir,
                      title=f'Transition A{si} → A{sf} (Cells)',
                      preferred='adaptive', max_cells=5000)
        ax_j = fig.add_subplot(gs[1, 1])
        _safe_heatmap(pl, adata_seacell, si, sf, ax_j, fig_dir,
                      title=f'Transition A{si} → A{sf} (Metacells)',
                      preferred='logistic', max_cells=None)

        for ext in ('pdf', 'png'):
            fig.savefig(f'{FIG_DIR}/larry_transition_{si}_to_{sf}_C_D_heatmaps.{ext}',
                        dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'  wrote larry_transition_{si}_to_{sf}_C_D_heatmaps (+ td_genes CSVs)')

    print(f'\nTransition figures + gene lists in {FIG_DIR}')


def main_light():
    print('Loading obs + X_umap (no expression matrix)...')
    df = load()
    print(f'  cells: {len(df)} | attractors: {df["attractor"].nunique()} | '
          f'anno: {df["anno"].nunique()} | '
          f'land range: {np.nanmin(df["land"]):.2f}-{np.nanmax(df["land"]):.2f}')
    fig1_heatmap(df)
    fig2_umaps(df)
    fig3_hist(df)
    fig4_dotplot(df)
    print(f'Done. Figures in {FIG_DIR}')


def main():
    ap = argparse.ArgumentParser(description='LARRY supplementary figures')
    ap.add_argument('--mode', choices=['light', 'transitions', 'all'], default='light',
                    help="'light' = cell-level summary figures (login-node safe); "
                         "'transitions' = per-transition heatmaps/plots/gene lists "
                         "(needs pyMuTrans + the 4.5 GB cell file, run on a compute node); "
                         "'all' = both.")
    args = ap.parse_args()
    if args.mode in ('light', 'all'):
        main_light()
    if args.mode in ('transitions', 'all'):
        generate_transition_figures()


if __name__ == '__main__':
    main()
