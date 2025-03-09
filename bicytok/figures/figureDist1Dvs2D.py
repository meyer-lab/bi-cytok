"""
Generate plots to compare 1D and 2D distance metrics, which should match.

Data Import:
- The CITE-seq dataframe (`importCITE`)

Parameters:
- targCell: cell type whose selectivity will be maximized
- receptors_of_interest: list of receptors to be analyzed
- sample_size: number of cells to sample for analysis
    (if greater than available cells, will use all)
- cell_categorization: column name in CITE-seq dataframe for cell type categorization

Outputs:
- Plots scatter plots comparing 1D and 2D distance metrics
- Each plot is labeled with the R^2 value of the comparison
"""

import numpy as np

from ..distance_metric_funcs import KL_EMD_1D, KL_EMD_2D
from ..imports import importCITE, sample_receptor_abundances
from .common import getSetup


def makeFigure():
    ax, f = getSetup((10, 5), (1, 2))

    # Parameters
    targCell = "Treg"
    receptors_of_interest = [
        "CD25",
        "CD4-1",
        "CD27",
        "CD4-2",
        "CD278",
        "CD122",
        "CD28",
        "TCR-2",
        "TIGIT",
        "TSLPR",
    ]
    sample_size = 100
    cell_categorization = "CellType2"


    CITE_DF = importCITE()

    assert targCell in CITE_DF[cell_categorization].unique()

    epitopesDF = CITE_DF[receptors_of_interest + [cell_categorization]]
    epitopesDF = epitopesDF.rename(columns={cell_categorization: "Cell Type"})
    sampleDF = sample_receptor_abundances(
        CITE_DF=epitopesDF,
        numCells=sample_size,
        targCellType=targCell,
    )
    rec_abundances = sampleDF[receptors_of_interest].to_numpy()

    target_mask = (sampleDF["Cell Type"] == targCell).to_numpy()
    off_target_mask = ~target_mask

    KL_div_vals_1D, EMD_vals_1D = KL_EMD_1D(
        rec_abundances, target_mask, off_target_mask
    )

    KL_div_vals_2D, EMD_vals_2D = KL_EMD_2D(
        rec_abundances, target_mask, off_target_mask, calc_1D=True
    )
    KL_div_vals_2D = np.diag(KL_div_vals_2D)
    EMD_vals_2D = np.diag(EMD_vals_2D)

    # EMD values have exact linear scaling between 1D and 2D based on Euclidean
    #   distance. KL divergence values have some non-linear scaling. The EMD
    #   scaling factor does a pretty good job of approximating the KL divergence.
    EMD_vals_2D_scaled = EMD_vals_2D * 2**0.5 / 2
    KL_div_vals_2D_scaled = KL_div_vals_2D * 2**0.5 / 2

    EMD_r2 = 1 - np.sum((EMD_vals_1D - EMD_vals_2D_scaled) ** 2) / np.sum(
        (EMD_vals_1D - np.mean(EMD_vals_1D)) ** 2
    )
    KL_div_r2 = 1 - np.sum((KL_div_vals_1D - KL_div_vals_2D_scaled) ** 2) / np.sum(
        (KL_div_vals_1D - np.mean(KL_div_vals_1D)) ** 2
    )

    # Scatter plot KL values
    ax[0].scatter(KL_div_vals_2D, KL_div_vals_1D, color="b")
    ax[0].set_title(f"KL Divergence Values\nR^2: {KL_div_r2:.2f}")
    ax[0].set_xlabel("2D KL Divergence")
    ax[0].set_ylabel("1D KL Divergence")

    # Scatter plot EMD values
    ax[1].scatter(EMD_vals_2D_scaled, EMD_vals_1D, color="g")
    ax[1].set_title(f"EMD Values\nR^2: {EMD_r2:.2f}")
    ax[1].set_xlabel("2D EMD")
    ax[1].set_ylabel("1D EMD")

    return f
