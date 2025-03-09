"""
Figure file to visualize the relationship between KL divergence and EMD metrics
across all receptors, highlighting outliers where metrics significantly disagree.

Data Import:
- The CITE-seq dataframe (`importCITE`)

Parameters:
- targCell: cell type whose receptor distributions will be compared to off-target cells
- sample_size: number of cells to sample for analysis
- cell_categorization: column name for cell type classification

Outputs:
- Scatter plot of KL divergence vs EMD for all receptors
- Highlighted and labeled outlier receptors
- Summary statistics on the correlation between these metrics
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.linear_model import LinearRegression

from ..distance_metric_funcs import KL_EMD_1D
from ..imports import importCITE, sample_receptor_abundances
from .common import getSetup

path_here = Path(__file__).parent.parent

plt.rcParams["svg.fonttype"] = "none"


def makeFigure():
    ax, f = getSetup((10, 8), (1, 1))
    ax = ax[0]

    # Parameters
    targCell = "Treg"
    sample_size = 100
    cell_categorization = "CellType2"

    # Load and prepare data
    CITE_DF = importCITE()

    # Ensure target cell exists in the dataset
    assert (
        targCell in CITE_DF[cell_categorization].unique()
    )

    # Sample cells for analysis
    epitopes = [
        col
        for col in CITE_DF.columns
        if col not in ["CellType1", "CellType2", "CellType3"]
    ]
    epitopesDF = CITE_DF[epitopes + [cell_categorization]]
    epitopesDF = epitopesDF.rename(columns={cell_categorization: "Cell Type"})
    sampleDF = sample_receptor_abundances(
        CITE_DF=epitopesDF,
        numCells=min(sample_size, epitopesDF.shape[0]),
        targCellType=targCell,
    )

    # Create target and off-target masks
    targ_mask = (sampleDF["Cell Type"] == targCell).to_numpy()
    off_targ_mask = ~targ_mask

    # Filter out any columns with all zero values
    filtered_sampleDF = sampleDF[sampleDF.columns[~sampleDF.columns.isin(["Cell Type"])]]
    
    # Filter columns with all zeros in off-target cells
    off_target_zeros = filtered_sampleDF.loc[off_targ_mask].apply(lambda col: (col == 0).all())
    filtered_sampleDF = filtered_sampleDF.loc[:, ~off_target_zeros]
    receptor_columns = filtered_sampleDF.columns

    # Get receptor abundances
    rec_abundances = filtered_sampleDF.to_numpy()

    # Calculate KL divergence and EMD for all receptors
    KL_div_vals, EMD_vals = KL_EMD_1D(rec_abundances, targ_mask, off_targ_mask)

    # Create DataFrame with results, removing NaN values
    results_df = pd.DataFrame(
        {"Receptor": receptor_columns, "KL_Divergence": KL_div_vals, "EMD": EMD_vals}
    ).dropna()

    # Calculate correlation between KL divergence and EMD
    correlation, _ = stats.pearsonr(
        results_df["KL_Divergence"], results_df["EMD"]
    )
    print(f"Pearson correlation: {correlation:.4f}")

    # Fit a linear regression
    reg_model = LinearRegression()
    X = results_df["EMD"].values.reshape(-1, 1)
    y = results_df["KL_Divergence"].values
    reg_model.fit(X, y)
    r_squared = reg_model.score(X, y)
    slope = reg_model.coef_[0]
    intercept = reg_model.intercept_

    # Calculate residuals
    results_df["Predicted_KL"] = intercept + slope * results_df["EMD"]
    results_df["Residual"] = results_df["KL_Divergence"] - results_df["Predicted_KL"]
    results_df["Abs_Residual"] = np.abs(results_df["Residual"])

    # Define outliers based on residuals
    # High KL, Low EMD: Positive residuals (KL higher than predicted from EMD)
    # Low KL, High EMD: Negative residuals (KL lower than predicted from EMD)
    residual_threshold = 1.5 * np.std(results_df["Residual"])

    results_df["Outlier_Type"] = "Normal"
    results_df.loc[results_df["Residual"] > residual_threshold, "Outlier_Type"] = (
        "High KL, Low EMD"
    )
    results_df.loc[results_df["Residual"] < -residual_threshold, "Outlier_Type"] = (
        "Low KL, High EMD"
    )

    # Get top outliers in each category
    high_kl_outliers = results_df[
        results_df["Outlier_Type"] == "High KL, Low EMD"
    ].nlargest(5, "Abs_Residual")
    low_kl_outliers = results_df[
        results_df["Outlier_Type"] == "Low KL, High EMD"
    ].nlargest(5, "Abs_Residual")

    # Create a color map for the scatter plot
    color_map = {
        "Normal": "gray",
        "High KL, Low EMD": "red",
        "Low KL, High EMD": "blue",
    }
    colors = [color_map[t] for t in results_df["Outlier_Type"]]

    # Plot the scatter plot
    scatter = ax.scatter(
        results_df["EMD"],
        results_df["KL_Divergence"],
        alpha=0.6,
        c=colors,
        s=30,
        edgecolors="k",
        linewidths=0.5,
    )

    # Add regression line
    x_range = np.linspace(min(results_df["EMD"]), max(results_df["EMD"]), 100)
    y_pred = intercept + slope * x_range
    ax.plot(x_range, y_pred, "k--", label=f"Linear fit (R²={r_squared:.2f})")

    # Label the outliers
    for _, row in pd.concat([high_kl_outliers, low_kl_outliers]).iterrows():
        ax.annotate(
            row["Receptor"],
            xy=(row["EMD"], row["KL_Divergence"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )

    # Add legend for outlier types
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markersize=10,
            label=label,
        )
        for label, color in color_map.items()
    ]
    ax.legend(handles=handles, title="Receptor Categories", loc="lower right")

    # Add a text box with correlation information
    stats_text = (
        f"Pearson correlation: {correlation:.4f}\n"
        f"R²: {r_squared:.4f}\n"
        f"Slope: {slope:.4f}\n"
        f"# Receptors: {len(results_df)}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        fontsize=9,
    )

    # Add summary of outliers at the bottom of the figure
    outlier_summary = (
        f"High KL, Low EMD outliers ({len(high_kl_outliers)} shown): {', '.join(high_kl_outliers['Receptor'])}\n"
        f"Low KL, High EMD outliers ({len(low_kl_outliers)} shown): {', '.join(low_kl_outliers['Receptor'])}"
    )
    plt.figtext(
        0.5,
        0.01,
        outlier_summary,
        ha="center",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Set titles and labels
    ax.set_title(
        f"KL Divergence vs EMD for {targCell} vs Off-target Cells", fontsize=12
    )
    ax.set_xlabel("Earth Mover's Distance (EMD)", fontsize=11)
    ax.set_ylabel("KL Divergence", fontsize=11)

    # Set axis properties
    ax.grid(True, linestyle="--", alpha=0.7)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    return f
