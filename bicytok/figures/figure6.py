"""
Generates a heatmap visualizing the Earth Mover's Distance (EMD)
between selected receptors (CD25 and CD35) for a target cell
type ("Treg") compared to off-target populations

Data Import:
- Loads the CITE-seq dataset using `importCITE`
    and samples the first 1000 rows for analysis.
- Identifies non-marker columns (`CellType1`, `CellType2`,
    `CellType3`, `Cell`) and filters out these columns
    to retain only the marker (receptor) columns for analysis.

Receptor Selection:
- Filters the marker dataframe to include only columns
    related to the receptors of interest, specifically
    `"CD25"` and `"CD35"`, focusing on these markers
    for the calculation of EMD.

Target and Off-Target Cell Definition:
- Creates a binary array for on-target cells based on
    the specified target cell type (`"Treg"`).
- Defines off-target cell populations using the
    `offTargState` parameter:
    - `offTargState = 0`: All non-memory Tregs.
    - `offTargState = 1`: All non-Tregs.
    - `offTargState = 2`: Only naive Tregs.

EMD Calculation:
- Computes an Earth Mover's Distance (EMD) matrix using
    the `EMD_2D` function to measure the dissimilarity
    between on-target ("Treg") and off-target cell distributions
    for the selected receptors (CD25 and CD35).
- Constructs a DataFrame (`df_recep`) to store the computed
    EMD values, with rows and columns labeled by the
    receptors of interest.

Visualization:
- Generates a heatmap of the EMD matrix using Seaborn's
    `heatmap` function.
- The heatmap uses a "bwr" color map to visually represent
    the EMD values, with annotations to display specific values.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

from ..distance_metric_funcs import KL_EMD_2D
from ..imports import importCITE
from ..selectivity_funcs import sample_receptor_abundances
from .common import getSetup

path_here = Path(__file__).parent.parent


def makeFigure():
    ax, f = getSetup((10, 5), (1, 2))
    np.random.seed(42)

    targCell = "Treg"
    offTargState = 1
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
    offTargCells = cellTypes[cellTypes != targCell]
    cell_categorization = "CellType2"

    assert any(np.array([0, 1, 2]) == offTargState)

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
