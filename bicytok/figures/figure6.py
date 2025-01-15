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

import numpy as np
import pandas as pd
import seaborn as sns

from ..distance_metric_funcs import KL_EMD_2D
from ..imports import importCITE
from .common import getSetup


def makeFigure():
    ax, f = getSetup((10, 5), (1, 2))

    targCell = "Treg"
    offTargState = 1
    receptors_of_interest = ["CD25", "CD35"]

    assert any(np.array([0, 1, 2]) == offTargState)

    CITE_DF = importCITE()
    # CITE_DF = CITE_DF.head(1000)

    # Define non-marker columns
    non_marker_columns = ["CellType1", "CellType2", "CellType3", "Cell"]
    marker_columns = CITE_DF.columns[~CITE_DF.columns.isin(non_marker_columns)]
    markerDF = CITE_DF.loc[:, marker_columns]

    # Further filter to include only columns related to CD25 and CD35
    filtered_markerDF = markerDF.loc[
        :, markerDF.columns.str.fullmatch("|".join(receptors_of_interest), case=False)
    ]

    on_target = (CITE_DF["CellType2"] == targCell).to_numpy()
    off_target_conditions = {
        0: (CITE_DF["CellType3"] != targCell),  # All non-memory Tregs
        1: (
            (CITE_DF["CellType2"] != "Treg") & (CITE_DF["CellType2"] != targCell)
        ),  # All non-Tregs
        2: (CITE_DF["CellType3"] == "Treg Naive"),  # Naive Tregs
    }
    off_target = off_target_conditions[offTargState].to_numpy()

    rec_abundances = filtered_markerDF.to_numpy()

    KL_div_vals, EMD_vals = KL_EMD_2D(rec_abundances, on_target, off_target)

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
