import numpy as np

from ..distanceMetricFuncs import KL_EMD_1D
from ..imports import importCITE
from .common import getSetup


def makeFigure():
    """
     Generates horizontal bar charts to visualize the top 5 markers with the highest KL Divergence and Earth Mover's Distance (EMD) values, comparing target and off-target cell distributions using CITE-seq data.

     Data Import:
    - Imports the CITE-seq dataframe (`importCITE`) and the plotting setup (`getSetup`).
    - Defines a target cell type (default: "Treg") and an off-target state (`offTargState`), specifying which cells are considered "off-target".

     Off-Target State Definitions:
    - Allows the selection of different off-target conditions using `offTargState`:
      - `offTargState = 0`: All non-memory Tregs.
      - `offTargState = 1`: All non-Tregs.
      - `offTargState = 2`: Only naive Tregs.

      KL Divergence and EMD Calculation**:
    - Computes the 1D KL divergence and EMD for each marker between the target and off-target cell distributions using `KL_EMD_1D`.
    - Returns two arrays: one with KL divergence values and one with EMD values.

    Identifies the top 5 markers with the highest KL divergence and the top 5 markers with the highest EMD.
    - Plots horizontal bar charts for these top markers:
      - **KL Divergence Plot**: Top 5 markers sorted by KL divergence.
      - **EMD Plot**: Top 5 markers sorted by EMD.
    - Each plot is labeled with marker names on the y-axis and their respective values (KL or EMD) on the x-axis.
    """
    
    ax, f = getSetup((8, 8), (1, 2))

    targCell = "Treg"
    offTargState = 0

    CITE_DF = importCITE()

    # Filter out non-marker columns
    non_marker_columns = ["CellType1", "CellType2", "CellType3", "Cell"]
    marker_columns = CITE_DF.columns[~CITE_DF.columns.isin(non_marker_columns)]
    markerDF = CITE_DF.loc[:, marker_columns]

    on_target = (CITE_DF["CellType3"] == targCell).to_numpy()

    off_target_conditions = {
        0: (CITE_DF["CellType3"] != targCell),  # All non-memory Tregs
        1: (CITE_DF["CellType2"] != "Treg"),  # All non-Tregs
        2: (CITE_DF["CellType3"] == "Treg Naive"),  # Naive Tregs
    }

    if offTargState in off_target_conditions:
        off_target_mask = off_target_conditions[offTargState].to_numpy()
    else:
        raise ValueError("Invalid offTargState value. Must be 0, 1, or 2.")

    recAbundances = markerDF.to_numpy()

    KL_values, EMD_values = KL_EMD_1D(recAbundances, on_target, off_target_mask)

    top_5_KL_indices = np.argsort(np.nan_to_num(KL_values))[-5:]
    top_5_EMD_indices = np.argsort(np.nan_to_num(EMD_values))[-5:]

    # Plot KL values
    ax[0].barh(
        markerDF.columns[top_5_KL_indices], KL_values[top_5_KL_indices], color="b"
    )
    ax[0].set_title("Top 5 KL Divergence Values")
    ax[0].set_xlabel("KL Divergence")
    ax[0].invert_yaxis()

    # Plot EMD values
    ax[1].barh(
        markerDF.columns[top_5_EMD_indices], EMD_values[top_5_EMD_indices], color="g"
    )
    ax[1].set_title("Top 5 EMD Values")
    ax[1].set_xlabel("EMD Value")
    ax[1].invert_yaxis()

    return f
