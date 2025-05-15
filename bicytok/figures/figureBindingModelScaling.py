"""
Generates a scatter plot that shows the effect of varying the scaling factor of raw
    receptor counts on the optimal selectivity of different receptors.

Data Import:
- The CITE-seq dataframe (`importCITE`)

Parameters:
- targCell: cell type whose selectivity will be maximized
- sample_size: number of cells to sample from the CITE-seq dataframe
- cell_categorization: column name in the CITE-seq dataframe that categorizes cells
- model_valencies: valencies each receptor's ligand in the model molecule
- dose: dose of the model molecule
- signal_receptor: receptor that is the target of the model molecule
- target_receptors: list of receptors to be tested for selectivity
- affinity_bounds: optimization bounds for the affinities of the receptors

Outputs:
- A plot showing the selectivity of each receptor as a function of the scaling factor
    applied to its raw counts
- The plot includes a dashed line indicating the selectivity of the receptor
    without any scaling
"""

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

from ..imports import importCITE, sample_receptor_abundances
from ..selectivity_funcs import optimize_affs
from .common import getSetup

path_here = Path(__file__).parent.parent


def makeFigure():
    ax, f = getSetup((15, 5), (1, 3))

    # Parameters
    targCell = "Treg"
    sample_size = 500
    cell_categorization = "CellType2"
    model_valencies = np.array([[(2), (2)]])
    dose = 10e-2
    signal_receptor = "CD122"
    target_receptors = ["CD25", "CD4-1", "CD27", "TIGIT"]
    affinity_bounds = (3, 13)
    num_conv_factors = 5

    CITE_DF = importCITE()

    epitopes_all = [
        col
        for col in CITE_DF.columns
        if col not in ["CellType1", "CellType2", "CellType3"]
    ]
    epitopesDF = CITE_DF[epitopes_all + [cell_categorization]]
    epitopesDF = epitopesDF.rename(columns={cell_categorization: "Cell Type"})
    sampleDF = sample_receptor_abundances(
        CITE_DF=epitopesDF,
        numCells=sample_size,
        targCellType=targCell,
        convert=False,
    )

    filterDF = pd.DataFrame(
        {
            signal_receptor: sampleDF[signal_receptor],
            "Cell Type": sampleDF["Cell Type"],
        }
    )
    for receptor in target_receptors:
        filterDF[receptor] = sampleDF[receptor]

    receptors_of_interest = [
        col for col in filterDF.columns if col not in ["Cell Type"]
    ]

    on_target_mask = (filterDF["Cell Type"] == targCell).to_numpy()
    off_target_mask = ~on_target_mask

    # Calculate raw selectivity baseline
    raw_selectivities = []
    raw_signal_affs = []
    raw_target_affs = []
    for receptor in receptors_of_interest:
        targRecs = filterDF[[signal_receptor, receptor]].to_numpy()
        offTargRecs = filterDF[[signal_receptor, receptor]].to_numpy()
        optSelec, optAffs = optimize_affs(
            targRecs=targRecs[on_target_mask],
            offTargRecs=offTargRecs[off_target_mask],
            dose=dose,
            valencies=model_valencies,
            bounds=affinity_bounds,
        )
        raw_selectivities.append(1 / optSelec)
        raw_signal_affs.append(optAffs[0])
        raw_target_affs.append(optAffs[1])

    # Calculate selectivity for each conversion factor
    conversion_factors = np.logspace(-2, 8, num=num_conv_factors)
    full_selectivities = []
    full_signal_affs = []
    full_target_affs = []
    for receptor in receptors_of_interest:
        per_rec_selectivities = []
        per_rec_signal_affs = []
        per_rec_target_affs = []
        for conv_fact in conversion_factors:
            test_DF = filterDF.copy()
            test_DF[receptor] = test_DF[receptor] * conv_fact
            targRecs = test_DF[[signal_receptor, receptor]].to_numpy()
            offTargRecs = test_DF[[signal_receptor, receptor]].to_numpy()
            optSelec, optAffs = optimize_affs(
                targRecs=targRecs[on_target_mask],
                offTargRecs=offTargRecs[off_target_mask],
                dose=dose,
                valencies=model_valencies,
                bounds=affinity_bounds,
            )
            selectivity = 1 / optSelec
            per_rec_selectivities.append(selectivity)
            per_rec_signal_affs.append(optAffs[0])
            per_rec_target_affs.append(optAffs[1])
        full_selectivities.append(per_rec_selectivities)
        full_signal_affs.append(per_rec_signal_affs)
        full_target_affs.append(per_rec_target_affs)

    # Plot selectivities against conversion factors
    palette = sns.color_palette("colorblind", n_colors=len(receptors_of_interest))
    for i, receptor in enumerate(receptors_of_interest):
        ax[0].plot(
            conversion_factors,
            full_selectivities[i],
            marker="o",
            label=receptor,
            color=palette[i],
        )
        ax[0].axhline(
            y=raw_selectivities[i], linestyle="--", color=palette[i], alpha=0.7
        )
    ax[0].axvline(x=10**0, linestyle="--", color="black", alpha=0.5)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Conversion Factor")
    ax[0].set_ylabel("Selectivity")
    ax[0].set_title("Receptor Selectivity vs Conversion Factor")
    ax[0].legend(title="Receptor")

    # Plot the signal affinities against conversion factors
    palette = sns.color_palette("colorblind", n_colors=len(receptors_of_interest))
    for i, receptor in enumerate(receptors_of_interest):
        ax[1].plot(
            conversion_factors,
            full_signal_affs[i],
            marker="o",
            label=receptor,
            color=palette[i],
        )
        ax[1].axhline(y=raw_signal_affs[i], linestyle="--", color=palette[i], alpha=0.7)
    ax[1].axvline(x=10**0, linestyle="--", color="black", alpha=0.5)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("Conversion Factor")
    ax[1].set_ylabel("Signal Receptor Affinity")
    ax[1].set_title("Signal Receptor Affinity vs Conversion Factor")
    ax[1].legend(title="Receptor")

    # Plot the target affinities against conversion factors
    palette = sns.color_palette("colorblind", n_colors=len(receptors_of_interest))
    for i, receptor in enumerate(receptors_of_interest):
        ax[2].plot(
            conversion_factors,
            full_target_affs[i],
            marker="o",
            label=receptor,
            color=palette[i],
        )
        ax[2].axhline(y=raw_target_affs[i], linestyle="--", color=palette[i], alpha=0.7)
    ax[2].axvline(x=10**0, linestyle="--", color="black", alpha=0.5)
    ax[2].set_xscale("log")
    ax[2].set_xlabel("Conversion Factor")
    ax[2].set_ylabel("Target Receptor Affinity")
    ax[2].set_title("Target Receptor Affinity vs Conversion Factor")
    ax[2].legend(title="Receptor")

    return f
