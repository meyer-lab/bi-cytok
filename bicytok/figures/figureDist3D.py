"""
Generates heatmaps visualizing the 2D EMD and KL Divergence
    of a given set of receptor scRNA-seq data

Data Import:
- The CITE-seq dataframe (`importCITE`)
- Reads a list of epitopes from a CSV file (`epitopeList.csv`)

Parameters:
- targCell: cell type whose selectivity will be maximized
- receptors_of_interest: list of receptors to be analyzed
- sample_size: number of cells to sample for analysis
    (if greater than available cells, will use all)
- cellTypes: Array of all relevant cell types
- cell_categorization: column name in CITE-seq dataframe for cell type categorization

Outputs:
- Generates heatmaps of the EMD and KL divergences between all relevant receptor
    pairs
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ..distance_metric_funcs import KL_EMD_3D
from ..imports import importCITE, sample_receptor_abundances
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
        "CD28",
        "CD45RB",
        "CD146",
        "TIGIT",
        "TSLPR",
        "GP130",
        "CD109",
    ]
    sample_size = 1000
    cellTypes = np.array(
        [
            "CD8 Naive",
            "NK",
            "CD8 TEM",
            "CD4 Naive",
            "CD4 CTL",
            "CD8 TCM",
            "CD8 Proliferating",
            "Treg",
        ]
    )
    cell_categorization = "CellType2"

    offTargCells = cellTypes[cellTypes != targCell]

    epitopesList = pd.read_csv(path_here / "data" / "epitopeList.csv")
    epitopes = list(epitopesList["Epitope"].unique())
    CITE_DF = importCITE()

    assert targCell in CITE_DF[cell_categorization].unique()

    epitopesDF = CITE_DF[epitopes + [cell_categorization]]
    epitopesDF = epitopesDF.loc[epitopesDF[cell_categorization].isin(cellTypes)]
    epitopesDF = epitopesDF.rename(columns={cell_categorization: "Cell Type"})
    sampleDF = sample_receptor_abundances(
        CITE_DF=epitopesDF,
        numCells=min(sample_size, epitopesDF.shape[0]),
        targCellType=targCell,
        offTargCellTypes=offTargCells,
    )
    filtered_sampleDF = sampleDF.loc[
        :,
        sampleDF.columns.str.fullmatch("|".join(receptors_of_interest), case=False),
    ]
    receptors_of_interest = filtered_sampleDF.columns

    on_target_mask = (sampleDF["Cell Type"] == targCell).to_numpy()
    off_target_mask = sampleDF["Cell Type"].isin(offTargCells).to_numpy()

    rec_abundances = filtered_sampleDF.to_numpy()

    KL_div_vals, EMD_vals = KL_EMD_3D(rec_abundances,
        on_target_mask,
        off_target_mask,
        calc_diags=True,
    )

    # Flatten the 3D arrays to 1D arrays
    KL_flat = KL_div_vals.flatten()
    EMD_flat = EMD_vals.flatten()

    # Get the indices of the top 10 values
    top_10_KL_indices = np.argsort(np.nan_to_num(KL_flat))[-10:]
    top_10_EMD_indices = np.argsort(np.nan_to_num(EMD_flat))[-10:]

    # Convert the flat indices back to 3D indices
    top_10_KL_combinations = np.unravel_index(top_10_KL_indices, KL_div_vals.shape)
    top_10_EMD_combinations = np.unravel_index(top_10_EMD_indices, EMD_vals.shape)

    # Get the receptor names for the top 10 combinations
    top_10_KL_receptors = [
        f"{receptors_of_interest[i]}-{receptors_of_interest[j]}-{receptors_of_interest[k]}"
        for i, j, k in zip(*top_10_KL_combinations, strict=False)
    ]
    top_10_EMD_receptors = [
        f"{receptors_of_interest[i]}-{receptors_of_interest[j]}-{receptors_of_interest[k]}"
        for i, j, k in zip(*top_10_EMD_combinations, strict=False)
    ]

    # Plot KL values
    ax[0].barh(
        top_10_KL_receptors,
        KL_flat[top_10_KL_indices],
        color="b",
    )
    ax[0].set_title("Top 10 KL Divergence Values")
    ax[0].set_xlabel("KL Divergence")
    ax[0].invert_yaxis()

    # Plot EMD values
    ax[1].barh(
        top_10_EMD_receptors,
        EMD_flat[top_10_EMD_indices],
        color="g",
    )
    ax[1].set_title("Top 10 EMD Values")
    ax[1].set_xlabel("EMD Value")
    ax[1].invert_yaxis()

    return f
