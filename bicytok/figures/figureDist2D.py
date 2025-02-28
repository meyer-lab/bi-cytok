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
import seaborn as sns

from ..distance_metric_funcs import KL_EMD_2D
from ..imports import importCITE, sample_receptor_abundances
from .common import getSetup

path_here = Path(__file__).parent.parent


def makeFigure():
    ax, f = getSetup((10, 5), (1, 2))
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

    KL_div_vals, EMD_vals = KL_EMD_2D(
        rec_abundances, on_target_mask, off_target_mask, calc_1D=False
    )

    EMD_matrix = np.tril(EMD_vals, k=0)
    EMD_matrix = EMD_matrix + EMD_matrix.T - np.diag(np.diag(EMD_matrix))
    KL_matrix = np.tril(KL_div_vals, k=0)
    KL_matrix = KL_matrix + KL_matrix.T - np.diag(np.diag(KL_matrix))

    df_EMD = pd.DataFrame(
        EMD_matrix, index=receptors_of_interest, columns=receptors_of_interest
    )
    df_KL = pd.DataFrame(
        KL_matrix, index=receptors_of_interest, columns=receptors_of_interest
    )

    # Visualize the EMD matrix with a heatmap
    sns.heatmap(
        df_EMD, cmap="bwr", annot=True, ax=ax[0], cbar=True, annot_kws={"fontsize": 16}
    )
    sns.heatmap(
        df_KL, cmap="bwr", annot=True, ax=ax[1], cbar=True, annot_kws={"fontsize": 16}
    )

    ax[0].set_title("EMD between: CD25 and CD35")
    ax[1].set_title("KL Divergence between: CD25 and CD35")

    return f
