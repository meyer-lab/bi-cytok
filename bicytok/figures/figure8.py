import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..distanceMetricFuncs import KL_EMD_2D
from ..imports import importCITE
from .common import getSetup


def makeFigure():
    """
     Generates a heatmap of KL divergence values for selected receptors (CD25, CD35) across a specified cell type.

     Data Import: Loads and filters CITE-seq data for the first 1000 rows.

     Receptor Selection:
    - Filters the marker dataframe to include only columns related to the receptors of interest, specifically
      `"CD25"` and `"CD35"`, for further analysis.

     Defines binary arrays for on-target cells (`Tregs`) and off-target cells based on the `offTargState` parameter:
      - `offTargState = 0`: All non-Tregs (including non-memory Tregs).
      - `offTargState = 1`: All non-Treg cells (excluding any type of Treg).
      - `offTargState = 2`: Naive Tregs only.

     Divergence Calculation:
    - Computes a KL divergence matrix using `KL_divergence_2D` to measure the dissimilarity between on-target
      ("Treg") and off-target cell distributions for the selected receptors (CD25 and CD35).
    - Constructs a DataFrame (`df_recep`) to store the computed KL divergence values, indexed and labeled by the receptors of interest.

     Visualization:
    - Generates a heatmap of the KL divergence matrix using Seaborn's `heatmap` function.
    - The heatmap uses a "bwr" color map to visually represent the divergence values, with annotations to display specific values.
    - Sets the title of the heatmap to indicate that it shows KL divergence between CD25 and CD35.

    """
    CITE_DF = importCITE()
    CITE_DF = CITE_DF.head(1000)

    ax, f = getSetup((40, 40), (1, 1))

    targCell = "Treg"
    offTargState = 0

    # Define non-marker columns
    non_marker_columns = ["CellType1", "CellType2", "CellType3", "Cell"]
    marker_columns = CITE_DF.columns[~CITE_DF.columns.isin(non_marker_columns)]
    markerDF = CITE_DF.loc[:, marker_columns]

    # Further filter to include only columns related to CD25 and CD35
    receptors_of_interest = ["CD25", "CD35"]
    filtered_markerDF = markerDF.loc[
        :, markerDF.columns.str.contains("|".join(receptors_of_interest), case=False)
    ]

    # Binary arrays for on-target and off-target cell types
    on_target = (CITE_DF["CellType3"] == targCell).to_numpy()

    # Define off-target conditions using a dictionary
    off_target_conditions = {
        0: (CITE_DF["CellType3"] != targCell),  # All non-memory Tregs
        1: (CITE_DF["CellType2"] != "Treg"),  # All non-Tregs
        2: (CITE_DF["CellType3"] == "Treg Naive"),  # Naive Tregs
    }

    if offTargState in off_target_conditions:
        off_target_mask = off_target_conditions[offTargState].to_numpy()
    else:
        raise ValueError("Invalid offTargState value. Must be 0, 1, or 2.")

    rec_abundances = filtered_markerDF.values

    KL_div_vals, EMD_vals = KL_EMD_2D(rec_abundances, on_target, off_target_mask)

    KL_div_matrix = np.triu(KL_div_vals, k=1) + np.triu(KL_div_vals.T, k=1)
    EMD_matrix = np.triu(EMD_vals, k=1) + np.triu(EMD_vals.T, k=1)

    df_KL = pd.DataFrame(
        KL_div_matrix, index=receptors_of_interest, columns=receptors_of_interest
    )
    df_EMD = pd.DataFrame(
        EMD_matrix, index=receptors_of_interest, columns=receptors_of_interest
    )

    # Visualize the EMD matrix with a heatmap
    sns.heatmap(
        df_EMD, cmap="bwr", annot=True, ax=ax, cbar=True, annot_kws={"fontsize": 16}
    )

    ax.set_title("KL Divergence between: CD25 and CD35")
    plt.show()

    return f
