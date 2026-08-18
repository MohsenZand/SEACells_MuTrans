#!/usr/bin/env python3
"""
Cell-level 4-panel UMAP + collection-day composition per attractor.

Same lightweight pattern as supp_larry_figures.py fig2: reads only obs +
obsm['X_umap'] from the cell-level MuTrans h5ad via h5py (no expression matrix).

Panels: Cell.type.clean | Time.point (collection day) | attractor | land

Also writes collection-day proportion per attractor (stacked bar + heatmap + CSV).

Run:  python LARRY/scripts/celllevel_umap.py
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
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import config

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 8

MUTRANS_DIR = config.PROCESSED_DATA_DIR / 'larry' / 'mutrans'
ORG_H5AD = MUTRANS_DIR / 'larry_seacells_org_mutrans.h5ad'
FIG_DIR = config.RESULTS_DIR / 'larry' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

DAY_COL = 'Time.point'
CELLTYPE_COL = 'Cell.type.clean'


def read_obs_column(obs, key, as_float=False):
    if key not in obs:
        raise KeyError(f"'{key}' not in h5ad obs; available: {list(obs.keys())}")
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
        return np.array([cats[i] if i >= 0 else 'NA' for i in codes])
    arr = g[:]
    if as_float:
        return arr.astype(float)
    if arr.dtype.kind in ('S', 'O', 'U'):
        return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr])
    return arr.astype(str)


def load_cell_table(h5ad_path=ORG_H5AD):
    with h5py.File(h5ad_path, 'r') as f:
        obs = f['obs']
        df = pd.DataFrame({
            CELLTYPE_COL: read_obs_column(obs, CELLTYPE_COL),
            DAY_COL: read_obs_column(obs, DAY_COL),
            'attractor': read_obs_column(obs, 'attractor'),
            'land': read_obs_column(obs, 'land', as_float=True),
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


def _day_order(vals):
    u = pd.unique(vals)
    try:
        return sorted(u, key=lambda x: float(x))
    except (ValueError, TypeError):
        return sorted(u)


def _style_umap_ax(ax):
    ax.set_xlabel('UMAP1', fontsize=9)
    ax.set_ylabel('UMAP2', fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _umap_categorical(ax, df, col, title, palette_name='tab20'):
    if col == 'attractor':
        cats = _attr_order(df[col])
    elif col == DAY_COL:
        cats = _day_order(df[col])
    else:
        cats = sorted(df[col].unique())
    palette = sns.color_palette(palette_name, len(cats))
    cmap = {c: palette[i] for i, c in enumerate(cats)}
    order = np.random.RandomState(0).permutation(len(df))
    colors = df[col].map(cmap).values
    ax.scatter(
        df['umap1'].values[order], df['umap2'].values[order],
        c=colors[order], s=1.2, alpha=0.8, rasterized=True, edgecolors='none',
    )
    handles = [
        plt.Line2D([0], [0], marker='o', linestyle='', markersize=5,
                   markerfacecolor=cmap[c], markeredgecolor='none', label=c)
        for c in cats
    ]
    ax.legend(handles=handles, fontsize=6, loc='center left',
              bbox_to_anchor=(1.02, 0.5), frameon=False, title=col, title_fontsize=7)
    ax.set_title(title, fontsize=12, fontweight='bold', loc='left')
    _style_umap_ax(ax)


def _umap_continuous(ax, df, values, title, cbar_label, cmap='Greys'):
    order = np.argsort(values)
    sc = ax.scatter(
        df['umap1'].values[order], df['umap2'].values[order],
        c=values[order], cmap=cmap, s=1.2, alpha=0.8,
        rasterized=True, edgecolors='none',
    )
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=12, fontweight='bold', loc='left')
    _style_umap_ax(ax)


def plot_four_panel_umap(df, fig_dir=FIG_DIR):
    stem = 'celllevel_umap_celltype_day_attractor_land'
    fig, axs = plt.subplots(1, 4, figsize=(28, 6.5))
    _umap_categorical(axs[0], df, CELLTYPE_COL, 'Cell.type.clean')
    _umap_categorical(axs[1], df, DAY_COL, 'collection day (Time.point)', palette_name='Set2')
    _umap_categorical(axs[2], df, 'attractor', 'attractor', palette_name='tab20')
    _umap_continuous(axs[3], df, df['land'].values, 'landscape (land)', 'land score')
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(fig_dir / f'{stem}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {stem}.{{png,pdf}}')


def plot_day_proportion_per_attractor(df, fig_dir=FIG_DIR):
    """Fraction of each attractor from each collection day (row-normalized)."""
    stem = 'celllevel_attractor_collection_day_proportion'
    attr_order = _attr_order(df['attractor'])
    day_order = _day_order(df[DAY_COL])
    ct = pd.crosstab(df['attractor'], df[DAY_COL])
    ct = ct.reindex(index=attr_order, columns=day_order, fill_value=0)
    prop = ct.div(ct.sum(axis=1), axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    bottom = np.zeros(len(prop))
    colors = sns.color_palette('Set2', len(day_order))
    x = np.arange(len(prop))
    for i, day in enumerate(day_order):
        vals = prop[day].values
        ax.bar(x, vals, bottom=bottom, label=f'day {day}', color=colors[i], width=0.85)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(prop.index)
    ax.set_ylabel('Fraction of attractor cells')
    ax.set_xlabel('MuTrans attractor')
    ax.set_title('Collection-day composition per attractor', fontweight='bold')
    ax.legend(title=DAY_COL, fontsize=8, title_fontsize=8)
    ax.set_ylim(0, 1)

    ax = axes[1]
    sns.heatmap(
        prop, cmap='Blues', vmin=0, vmax=1, annot=True, fmt='.2f',
        linewidths=0.4, linecolor='white',
        cbar_kws={'label': 'Fraction of attractor', 'shrink': 0.7},
        ax=ax,
    )
    ax.set_xlabel('collection day (Time.point)')
    ax.set_ylabel('MuTrans attractor')
    ax.set_title('Day proportion heatmap', fontweight='bold')

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(fig_dir / f'{stem}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {stem}.{{png,pdf}}')

    csv_path = fig_dir / 'attractor_collection_day_proportion.csv'
    prop.to_csv(csv_path)
    print(f'  wrote {csv_path}')
    return prop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5ad', default=str(ORG_H5AD))
    ap.add_argument('--fig-dir', default=str(FIG_DIR))
    args = ap.parse_args()
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading obs + X_umap from {args.h5ad} ...')
    df = load_cell_table(args.h5ad)
    print(f'  cells: {len(df):,}')
    print(f'  attractors: {df["attractor"].nunique()} | cell types: {df[CELLTYPE_COL].nunique()}')
    print(f'  collection days: {_day_order(df[DAY_COL].unique())}')

    plot_four_panel_umap(df, fig_dir)
    prop = plot_day_proportion_per_attractor(df, fig_dir)
    print('\nDay proportion per attractor (head):')
    print(prop.head().to_string())
    print(f'\nDone. Figures in {fig_dir}')


if __name__ == '__main__':
    main()
