# Transitional Dynamics with SEACells + MuTrans

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22002925.svg)](https://doi.org/10.5281/zenodo.22002925)

Reproducible analysis pipelines that identify transitional cell states and
transition-driver genes by aggregating single cells into **metacells** with
[SEACells](https://github.com/dpeerlab/SEACells) and modelling cellular dynamics
with [MuTrans](https://github.com/cliffzhou92/MuTrans-release).

Two datasets are analysed, each in its own self-contained folder that shares the
common analysis library in [`src/`](src/):

| Folder | Dataset | Description |
|--------|---------|-------------|
| [`ipsc_cardiomyocyte/`](ipsc_cardiomyocyte/) | GEO **GSE175634** | iPSC-derived cardiomyocyte differentiation |
| [`LARRY/`](LARRY/) | GEO **GSE140802** | LARRY lineage-traced hematopoiesis (NM trajectory) |

## Repository layout

```
.
├── src/                         # shared library (imported by both analyses)
│   ├── config.py                # paths & parameters (repo-relative; env-var inputs)
│   ├── transcendental.py        # TCS + transition-driver / meta-stable / hybrid gene calls
│   └── plotting.py              # UMAP / violin / flow-matrix / transcendental heatmaps
│
├── ipsc_cardiomyocyte/
│   ├── scripts/                 # 01_run_seacells -> 02_run_mutrans -> 03/04 figures; seacells_eval
│   └── notebooks/               # interactive equivalents of the scripts
│
├── LARRY/
│   └── scripts/                 # supp_larry (pipeline) + supp_larry_figures + celllevel_umap
│       └── supp_larry.sbatch    # SLURM submission for the SEACells+MuTrans fit
│
├── environment.yml
└── README.md
```

`data/` and `results/` are created at run time and are git-ignored.

## Setup

**1. Environment**
```bash
conda env create -f environment.yml
conda activate ipsc_mutrans_seacells
```

**2. Manual dependencies** (not available as plain conda/PyPI packages)
```bash
pip install git+https://github.com/dpeerlab/SEACells.git      # SEACells
git clone https://github.com/cliffzhou92/MuTrans-release       # MuTrans
export MUTRANS_PATH=/path/to/MuTrans-release                   # point config at it
```

**3. Data** — download the raw `.h5ad` files from GEO and either place them under
`data/raw/` or point the environment variables at them:
```bash
export IPSC_RAW_H5AD=/path/to/GSE175634_iPSC_CM.sct3k_reclustered.h5ad
export LARRY_RAW_H5AD=/path/to/larry_NM_trajectory_normedCounts.h5ad
```
All other paths are repo-relative (see [`src/config.py`](src/config.py)); nothing is
machine-specific.

## Running

Run every command from the repository root so `from src import ...` resolves.

### iPSC cardiomyocyte (GSE175634)
```bash
python ipsc_cardiomyocyte/scripts/01_run_seacells.py          # raw -> SEACell metacells
python ipsc_cardiomyocyte/scripts/02_run_mutrans.py           # metacells -> MuTrans + lineage CSVs
python ipsc_cardiomyocyte/scripts/03_generate_all_figures.py  # comprehensive multi-panel figures
python ipsc_cardiomyocyte/scripts/04_generate_individual_figures.py --si 3 --sf 4
```

### LARRY hematopoiesis (GSE140802)
```bash
# SEACells (92.5k cells -> 1.2k metacells) + MuTrans (K=14). Long fit -> use SLURM:
sbatch LARRY/scripts/supp_larry.sbatch          # edit account/partition/env first

# Cell-level summary figures (login-node safe; reads only obs + UMAP):
python LARRY/scripts/supp_larry_figures.py --mode light
python LARRY/scripts/celllevel_umap.py

# Per-transition heatmaps + transition-driver gene lists (needs MuTrans; compute node):
python LARRY/scripts/supp_larry_figures.py --mode transitions
```

## Method (shared library)

1. **SEACells** aggregates single cells into metacells.
2. **MuTrans** `dynamical_analysis` assigns each metacell an `attractor`, an
   `entropy`, and a `land` (landscape) score, and yields a transition matrix.
3. **Transcendental / transition analysis** ([`src/transcendental.py`](src/transcendental.py)):
   for a source→target attractor pair it computes a per-cell **Transition Cell
   Score (TCS)**, orders cells along it, and classifies genes into
   **TD** (transition-driver), **MS** (meta-stable) and **IH** (intermediate-hybrid)
   categories, rendered as the transition heatmaps.

## Citation

This code accompanies the manuscript:

> Yang, X. H.<sup>1,\*</sup>, Yu, F.<sup>1</sup>, Zand, M.<sup>2</sup>, Ai, H.<sup>1</sup>, Lou, T.<sup>1</sup>, & Moskowitz, I. P.<sup>1</sup>
> *TIPS anticipates cell fate from transient progenitor bottlenecks through
> network edge reweighting.*

<sup>1</sup> Department of Pediatrics, The University of Chicago, Chicago, IL 60637, USA
<sup>2</sup> Research Computing Center (RCC), The University of Chicago, Chicago, IL 60637, USA
<sup>\*</sup> Corresponding author (X. H. Yang)

Please cite the archived software release:

```bibtex
@software{yang_tips_2026,
  author    = {Yang, Xinan H. and Yu, Felix and Zand, Mohsen and
               Ai, Horatio and Lou, Tingjun and Moskowitz, Ivan P.},
  title     = {TIPS anticipates cell fate from transient progenitor bottlenecks
               through network edge reweighting},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22002925},
  url       = {https://doi.org/10.5281/zenodo.22002925}
}
```

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
