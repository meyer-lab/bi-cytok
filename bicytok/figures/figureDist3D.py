"""
Generates a figure that shows the receptor triplets that have the highest 3D KL
    divergence and EMD values.

Data Import:
- The CITE-seq dataframe (`importCITE`)

Parameters:
- targCell: cell type whose selectivity will be maximized
- receptors_of_interest: list of receptors to be analyzed
- sample_size: number of cells to sample for analysis
    (if greater than available cells, will use all)
- cell_categorization: column name in CITE-seq dataframe for cell type categorization

Outputs:
- Two bar plots showing the top 10 KL divergence and EMD values
"""

from pathlib import Path

import numpy as np

from ..distance_metric_funcs import KL_EMD_3D
from ..imports import filter_receptor_abundances, importCITE, sample_receptor_abundances
from .common import getSetup

path_here = Path(__file__).parent.parent


def makeFigure():
    ax, f = getSetup((12, 10), (2, 1))
    np.random.seed(42)

    targCell = "Treg"
    receptors_of_interest = [
        "CD25",
        "CD4-1",
        "CD27",
        "CD4-2",
        "CD278",
    ]
    sample_size = 100
    cell_categorization = "CellType2"

    CITE_DF = importCITE()

    assert targCell in CITE_DF[cell_categorization].unique()

    epitopesDF = CITE_DF[receptors_of_interest + [cell_categorization]]
    epitopesDF = epitopesDF.rename(columns={cell_categorization: "Cell Type"})
    sampleDF = sample_receptor_abundances(
        CITE_DF=epitopesDF,
        numCells=min(sample_size, epitopesDF.shape[0]),
        targCellType=targCell,
    )
    filtered_sampleDF = filter_receptor_abundances(sampleDF, targCell)

    on_target_mask = (filtered_sampleDF["Cell Type"] == targCell).to_numpy()
    off_target_mask = ~on_target_mask
    rec_abundances = filtered_sampleDF[receptors_of_interest].to_numpy()
    KL_div_vals, EMD_vals = KL_EMD_3D(
        rec_abundances,
        on_target_mask,
        off_target_mask,
        calc_diags=True,
    )

    # Flatten the 3D arrays to 1D arrays
    KL_flat = KL_div_vals.flatten()
    EMD_flat = EMD_vals.flatten()

    # Get the indices of the top 5 values
    top_KL_indices = np.argsort(np.nan_to_num(KL_flat))[-5:]
    top_EMD_indices = np.argsort(np.nan_to_num(EMD_flat))[-5:]

    # Convert the flat indices back to 3D indices
    top_KL_combinations = np.unravel_index(top_KL_indices, KL_div_vals.shape)
    top_EMD_combinations = np.unravel_index(top_EMD_indices, EMD_vals.shape)

    # Get the receptor names for the top 5 combinations
    top_KL_receptors = [
        f"{receptors_of_interest[i]}-{receptors_of_interest[j]}-{receptors_of_interest[k]}"
        for i, j, k in zip(*top_KL_combinations, strict=False)
    ]
    top_EMD_receptors = [
        f"{receptors_of_interest[i]}-{receptors_of_interest[j]}-{receptors_of_interest[k]}"
        for i, j, k in zip(*top_EMD_combinations, strict=False)
    ]

    # Plot KL values
    ax[0].barh(
        top_KL_receptors,
        KL_flat[top_KL_indices],
        color="b",
    )
    ax[0].set_title("Top 5 KL Divergence Values")
    ax[0].set_xlabel("KL Divergence")
    ax[0].invert_yaxis()

    # Plot EMD values
    ax[1].barh(
        top_EMD_receptors,
        EMD_flat[top_EMD_indices],
        color="g",
    )
    ax[1].set_title("Top 5 EMD Values")
    ax[1].set_xlabel("EMD Value")
    ax[1].invert_yaxis()

    return f
