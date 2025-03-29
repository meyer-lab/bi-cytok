"""
Figure file to visualize KL divergence and EMD between different Gaussian
    distributions.

Parameters:
- means: List of means for different Gaussian distributions
- sigmas: List of standard deviations for different Gaussian distributions
- target_configs: List of tuples with target distribution parameters to compare against
    the reference distribution
- n_samples: Number of samples to generate for each distribution

Outputs:
- Plots of reference and target Gaussian distributions
- KL divergence and EMD metrics calculated from sample data
- Comparison of metrics across different distribution parameters
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from ..distance_metric_funcs import KL_EMD_1D
from .common import getSetup

path_here = Path(__file__).parent.parent


def makeFigure():
    ax, f = getSetup((15, 10), (3, 3))
    np.random.seed(42)  # For reproducibility

    # Parmeters
    ref_mean = 10.0
    ref_sigma = 2.0
    off_target_configs = [
        ("Same Mean, Same Sigma", ref_mean, ref_sigma),
        ("Same Mean, Larger Sigma", ref_mean, 3.0),
        ("Same Mean, Even Larger Sigma", ref_mean, 4.0),
        ("Same Mean, Same Sigma", ref_mean, ref_sigma),
        ("Larger Mean, Same Sigma", 15.0, ref_sigma),
        ("Even Larger Mean, Same Sigma", 20.0, ref_sigma),
        ("Larger Mean, Same Sigma", 15.0, ref_sigma),
        ("Larger Mean, Larger Sigma", 15.0, 3.0),
        ("Larger Mean, Even Larger Sigma", 15.0, 4.0),
    ]
    n_samples = 1500

    for i, (title, target_mean, target_sigma) in enumerate(off_target_configs):
        # Generate samples from both distributions
        target_samples = np.random.normal(ref_mean, ref_sigma, n_samples)
        off_target_samples = np.random.normal(target_mean, target_sigma, n_samples)

        # Calculate KL divergence and EMD using the project's function
        combined_samples = np.vstack([target_samples, off_target_samples]).reshape(
            -1, 1
        )
        targ_mask = np.array([True] * n_samples + [False] * n_samples)
        off_targ_mask = ~targ_mask
        KL_div, EMD = KL_EMD_1D(
            combined_samples, targ_mask, off_targ_mask, filter_recs=False
        )

        # Define x values for plotting the PDFs
        x = np.linspace(0, 2 * max(ref_mean, target_mean), 1000)
        ref_pdf = stats.norm.pdf(x, ref_mean, ref_sigma)
        target_pdf = stats.norm.pdf(x, target_mean, target_sigma)

        # Plot the distributions
        ax[i].plot(
            x,
            ref_pdf,
            label=f"Reference N({ref_mean},{ref_sigma})",
            color="blue",
            linewidth=2,
        )
        ax[i].plot(
            x,
            target_pdf,
            label=f"Target N({target_mean},{target_sigma})",
            color="red",
            linewidth=2,
        )
        ax[i].fill_between(
            x,
            np.minimum(ref_pdf, target_pdf),
            color="purple",
            alpha=0.3,
            label="Overlapping Area",
        )

        # Add KL/EMD metrics to the plot
        kl_value = KL_div[0] if not np.isnan(KL_div[0]) else float("nan")
        emd_value = EMD[0] if not np.isnan(EMD[0]) else float("nan")

        ax[i].annotate(
            f"KL: {kl_value:.4f}\nEMD: {emd_value:.4f}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            va="top",
            bbox=dict(boxstyle="round", fc="w", alpha=0.8),
        )
        ax[i].set_title(title)
        ax[i].set_xlabel("Value")
        ax[i].set_ylabel("Density")
        ax[i].legend()
        ax[i].grid(True, linestyle="--", alpha=0.7)

    plt.suptitle(
        "Comparing KL Divergence and EMD for Gaussian Distributions", fontsize=16
    )

    return f
