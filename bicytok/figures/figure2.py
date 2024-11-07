from os.path import dirname

import numpy as np
from imports import importCITE

from ..distanceMetricFuncs import KL_EMD_1D
from .common import getSetup

path_here = dirname(dirname(__file__))


def makeFigure():
    """Figure file to generate 1D KL divergence and EMD for given cell type/subset."""
    ax, f = getSetup((8, 8), (1, 2))

    targCell = "Treg"
    offTargState = 0

    CITE_DF = importCITE()

    # Filter out non-marker columns
    non_marker_columns = ["CellType1", "CellType2", "CellType3", "Cell"]
    marker_columns = CITE_DF.columns[~CITE_DF.columns.isin(non_marker_columns)]
    markerDF = CITE_DF.loc[:, marker_columns]

    # Binary arrays for on-target and off-target cell types
    on_target = (CITE_DF["CellType3"] == targCell).astype(int)

    # Define off-target conditions using a dictionary
    off_target_conditions = {
        0: (CITE_DF["CellType3"] != targCell),  # All non-memory Tregs
        1: (CITE_DF["CellType2"] != "Treg"),  # All non-Tregs
        2: (CITE_DF["CellType3"] == "Treg Naive"),  # Naive Tregs
    }

    # Set off_target based on offTargState
    if offTargState in off_target_conditions:
        off_target = off_target_conditions[offTargState].astype(int)
    else:
        raise ValueError("Invalid offTargState value. Must be 0, 1, or 2.")

    KL_values, EMD_values = KL_EMD_1D(markerDF, on_target, off_target)

    top_5_KL_indices = np.argsort(KL_values)[-5:]
    top_5_EMD_indices = np.argsort(EMD_values)[-5:]

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
