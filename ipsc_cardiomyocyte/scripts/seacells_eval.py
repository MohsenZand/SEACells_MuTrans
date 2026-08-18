import os
import sys
import random
import scanpy as sc
import SEACells
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Add project root to path to allow 'from src import ...'
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src import config # Import paths and params

    
    
    
def main():
    
    ad = sc.read_h5ad(config.MUTRANS_AD_ORG)
    
    
    
    purity_df = SEACells.evaluate.compute_celltype_purity(ad, 'leiden_0.5')
    purity_col_name = purity_df.columns[0] # Grab the actual column name

    # Map the purity score back to the single-cell metadata
    # We match the 'SEACell' column in ad.obs to the index of purity_df
    ad.obs['seacell_purity_score'] = ad.obs['SEACell'].map(purity_df['leiden_0.5_purity'])

    # Ensure attractor is categorical for better plotting
    ad.obs['attractor'] = ad.obs['attractor'].astype('category')
    
    
    # ==========================================
    # FIGURE 1: Purity vs. Leiden Clusters
    # ==========================================
    plt.figure(figsize=(12, 6)) # Create a new, independent figure context

    # Plot
    ax1 = sns.boxplot(data=ad.obs, x='leiden_0.5', y='seacell_purity_score', palette='viridis')

    # Customization
    plt.title('SEACell Purity across Original Clusters (Leiden 0.5)', fontsize=14)
    plt.ylabel('Purity Score (Max=1.0)', fontsize=12)
    plt.xlabel('Leiden Cluster ID', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    sns.despine()

    # Set y-axis limits explicitly to show the full 0-1 range if desired
    plt.ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig('seacell_purity_leiden.png')
    

    # ==========================================
    # FIGURE 2: Purity vs. MuTrans Attractors
    # ==========================================
    plt.figure(figsize=(10, 6)) # Create a new, independent figure context

    # Plot
    # We use a different palette to distinguish it from the first plot
    ax2 = sns.boxplot(data=ad.obs, x='attractor', y='seacell_purity_score', palette='rocket')

    # Customization
    plt.title('SEACell Purity across MuTrans Attractors', fontsize=14)
    plt.ylabel('Purity Score (relative to Leiden 0.5)', fontsize=12)
    plt.xlabel('Attractor ID', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    sns.despine()

    # Set y-axis limits explicitly to show the full 0-1 range if desired
    plt.ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig('seacell_purity_attractors.png')
    
    
    


if __name__ == "__main__":
    main()