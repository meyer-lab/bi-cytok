"""
Figure file to visualize KL divergence and EMD between Gaussian distributions.

This figure demonstrates the theoretical properties of KL divergence and EMD by:
1. Plotting pairs of Gaussian distributions with varying parameters
2. Using the project's KL_EMD_1D function to calculate empirical metrics
3. Showing how distribution parameters affect these metrics

Parameters:
- means: List of means for different Gaussian distributions
- sigmas: List of standard deviations for different Gaussian distributions

Outputs:
- Plots of reference and target Gaussian distributions
- KL divergence and EMD metrics calculated from sample data
- Comparison of metrics across different distribution parameters
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from ..distance_metric_funcs import KL_EMD_1D
from .common import getSetup

path_here = Path(__file__).parent.parent

plt.rcParams["svg.fonttype"] = "none"


def makeFigure():
    """Generate figure showing Gaussian distributions and their KL/EMD metrics"""
    ax, f = getSetup((15, 10), (3, 3))
    
    # Define reference distribution parameters
    ref_mean = 10.0
    ref_sigma = 2.0
    
    # Define target distribution parameter sets to demonstrate various KL/EMD scenarios
    target_configs = [
        ("Same Mean, Same Sigma", 10.0, 2.0), 
        ("Same Mean, Larger Sigma", 10.0, 3.0),
        ("Same Mean, Even Larger Sigma", 10.0, 4.0),  
        ("Same Mean, Same Sigma", 10.0, 2.0), 
        ("Larger Mean, Same Sigma", 15.0, 2.0), 
        ("Even Larger Mean, Same Sigma", 20.0, 2.0), 
        ("Larger Mean, Same Sigma", 15.0, 2.0), 
        ("Larger Mean, Larger Sigma", 15.0, 3.0),
        ("Larger Mean, Even Larger Sigma", 15.0, 4.0)

    ]
    
    np.random.seed(42)  # For reproducibility
    n_samples = 1500
    
    for i, (title, target_mean, target_sigma) in enumerate(target_configs):
        # Generate samples from both distributions
        ref_samples = np.random.normal(ref_mean, ref_sigma, n_samples)
        target_samples = np.random.normal(target_mean, target_sigma, n_samples)
        
        # Create masks for KL_EMD_1D (all ref samples are "target", all target samples are "off-target")
        combined_samples = np.vstack([ref_samples, target_samples]).reshape(-1, 1)
        targ_mask = np.array([True] * n_samples + [False] * n_samples)
        off_targ_mask = ~targ_mask
        
        # Calculate KL divergence and EMD using the project's function
        KL_div, EMD = KL_EMD_1D(combined_samples, targ_mask, off_targ_mask, filter_recs=False)
        
        # Define x values for plotting the PDFs
        x = np.linspace(0, 2 * max(ref_mean, target_mean), 1000)
        ref_pdf = stats.norm.pdf(x, ref_mean, ref_sigma)
        target_pdf = stats.norm.pdf(x, target_mean, target_sigma)
        
        # Plot the distributions
        ax[i].plot(x, ref_pdf, label=f"Reference N({ref_mean},{ref_sigma})", color='blue', linewidth=2)
        ax[i].plot(x, target_pdf, label=f"Target N({target_mean},{target_sigma})", color='red', linewidth=2)
        
        # Add shaded areas to visualize differences in distributions
        ax[i].fill_between(x, np.minimum(ref_pdf, target_pdf), color='purple', alpha=0.3, 
                          label="Overlapping Area")
        
        # Add KL/EMD metrics to the plot
        kl_value = KL_div[0] if not np.isnan(KL_div[0]) else float('nan')
        emd_value = EMD[0] if not np.isnan(EMD[0]) else float('nan')
        
        # Annotate with divergence metrics
        ax[i].annotate(f"KL: {kl_value:.4f}\nEMD: {emd_value:.4f}", 
                      xy=(0.05, 0.95),
                      xycoords='axes fraction',
                      va='top',
                      bbox=dict(boxstyle="round", fc="w", alpha=0.8))
        
        # Set title and labels
        ax[i].set_title(title)
        ax[i].set_xlabel("Value")
        ax[i].set_ylabel("Density")
        ax[i].legend()
        ax[i].grid(True, linestyle='--', alpha=0.7)
    
    # Create summary table of metrics
    summary_data = {
        "Distribution Pair": [config[0] for config in target_configs],
        "Mean Diff": [abs(config[1] - ref_mean) for config in target_configs],
        "Sigma Ratio": [config[2] / ref_sigma for config in target_configs]
    }
    
    # Calculate KL and EMD for all configurations and add to summary
    KL_values = []
    EMD_values = []
    
    for _, target_mean, target_sigma in target_configs:
        # Generate samples and calculate metrics
        ref_samples = np.random.normal(ref_mean, ref_sigma, n_samples)
        target_samples = np.random.normal(target_mean, target_sigma, n_samples)
        
        combined_samples = np.vstack([ref_samples, target_samples]).reshape(-1, 1)
        targ_mask = np.array([True] * n_samples + [False] * n_samples)
        off_targ_mask = ~targ_mask
        
        KL, EMD = KL_EMD_1D(combined_samples, targ_mask, off_targ_mask, filter_recs=False)
        KL_values.append(KL[0] if not np.isnan(KL[0]) else float('nan'))
        EMD_values.append(EMD[0] if not np.isnan(EMD[0]) else float('nan'))
    
    summary_data["KL Divergence"] = KL_values
    summary_data["EMD"] = EMD_values
    
    # Convert to DataFrame for printing
    summary_df = pd.DataFrame(summary_data)
    print(summary_df)
    
    # Set overall figure title
    plt.suptitle("Comparing KL Divergence and EMD for Gaussian Distributions", fontsize=16)
    
    return f
