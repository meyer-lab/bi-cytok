"""
Figure file to visualize the comparison between a raw histogram and kernel density estimation.

This figure demonstrates:
1. A simple histogram of data from a Gaussian distribution
2. The same histogram with a kernel density estimation overlay

Parameters:
- n_samples: Number of samples from the Gaussian distribution (set to 500)
- bandwidth: Method for KDE bandwidth selection (set to "scott")

Outputs:
- Two plots side by side: histogram alone and histogram with KDE overlay
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.neighbors import KernelDensity

from .common import getSetup

path_here = Path(__file__).parent.parent

plt.rcParams["svg.fonttype"] = "none"


def makeFigure():
    """Generate figure showing a histogram and the same histogram with KDE overlay"""
    ax, f = getSetup((12, 6), (1, 2))
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define Gaussian distribution parameters
    mean = 5.0
    sigma = 1.0
    
    # Set sample size and bandwidth method
    n_samples = 500
    bandwidth_method = "scott"
    
    # Generate data
    data = np.random.normal(mean, sigma, n_samples)
    
    # Plot 1: Histogram only
    sns.histplot(
        data,
        ax=ax[0],
        color='skyblue',
        alpha=0.7,
        stat="density",
        label=f"Histogram (n={n_samples})"
    )
    
    # Set title and labels for first plot
    ax[0].set_title("Histogram of Gaussian Data")
    ax[0].set_xlabel("Value")
    ax[0].set_ylabel("Density")
    ax[0].legend()
    ax[0].grid(True, linestyle='--', alpha=0.5)
    
    # Plot 2: Histogram with KDE overlay
    sns.histplot(
        data,
        ax=ax[1],
        color='skyblue',
        alpha=0.7,
        stat="density",
        label=f"Histogram (n={n_samples})"
    )
    
    # Calculate and plot KDE using scott bandwidth parameter
    x = np.linspace(mean - 4*sigma, mean + 4*sigma, 1000)
    kde = KernelDensity(bandwidth=bandwidth_method).fit(data.reshape(-1, 1))
    kde_values = np.exp(kde.score_samples(x.reshape(-1, 1)))
    ax[1].plot(x, kde_values, 'r-', lw=2, label=f"KDE ({bandwidth_method})")
    
    # Set title and labels for second plot
    ax[1].set_title("Histogram with KDE Overlay")
    ax[1].set_xlabel("Value")
    ax[1].set_ylabel("Density")
    ax[1].legend()
    ax[1].grid(True, linestyle='--', alpha=0.5)
    
    # Ensure both plots have the same y-axis limits for better comparison
    y_max = max(ax[0].get_ylim()[1], ax[1].get_ylim()[1])
    ax[0].set_ylim(0, y_max)
    ax[1].set_ylim(0, y_max)
    
    # Set overall figure title
    plt.suptitle("Raw Histogram vs. Kernel Density Estimation", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return f


if __name__ == "__main__":
    fig = makeFigure()
    plt.show()