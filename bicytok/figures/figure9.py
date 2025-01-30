"""Generate plots to compare 1D and 2D distance metrics, which should match."""

import numpy as np

from ..distance_metric_funcs import KL_EMD_1D, KL_EMD_2D
from ..imports import importCITE
from .common import getSetup


def makeFigure():
    ax, f = getSetup((10, 5), (2, 2))
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

    assert any(np.array([0, 1, 2]) == offTargState)

    CITE_DF = importCITE()

    # Define non-marker columns
    non_marker_columns = ["CellType1", "CellType2", "CellType3", "Cell"]
    marker_columns = CITE_DF.columns[~CITE_DF.columns.isin(non_marker_columns)]
    markerDF = CITE_DF.loc[:, marker_columns]
    filtered_markerDF = markerDF.loc[
        :,
        markerDF.columns.str.fullmatch("|".join(receptors_of_interest), case=False),
    ]
    receptors_of_interest = filtered_markerDF.columns

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

    # Randomly sample a subset of rows
    subset_indices = np.random.choice(
        len(on_target), size=min(sample_size, len(on_target)), replace=False
    )
    on_target = on_target[subset_indices]
    off_target = off_target[subset_indices]
    rec_abundances = rec_abundances[subset_indices]

    KL_div_vals_1D, EMD_vals_1D = KL_EMD_1D(rec_abundances, on_target, off_target)

    KL_div_vals_2D, EMD_vals_2D = KL_EMD_2D(
        rec_abundances, on_target, off_target, calc_1D=True
    )
    KL_div_vals_2D = np.diag(KL_div_vals_2D)
    EMD_vals_2D = np.diag(EMD_vals_2D)

    # Plot KL values
    ax[0].barh(
        receptors_of_interest,
        KL_div_vals_1D,
        color="b",
    )
    ax[0].set_title("1D KL Divergence Values")
    ax[0].set_xlabel("1D KL Divergence")
    ax[2].barh(
        receptors_of_interest,
        KL_div_vals_2D,
        color="b",
    )
    ax[2].set_title("2D KL Divergence Values")
    ax[2].set_xlabel("2D KL Divergence")

    # Plot EMD values
    ax[1].barh(
        receptors_of_interest,
        EMD_vals_1D,
        color="g",
    )
    ax[1].set_title("1D EMD Values")
    ax[1].set_xlabel("1D EMD")
    ax[3].barh(
        receptors_of_interest,
        EMD_vals_2D,
        color="g",
    )
    ax[3].set_title("2D EMD Values")
    ax[3].set_xlabel("2D EMD")

    return f
