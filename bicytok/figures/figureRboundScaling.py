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
from ..selectivity_funcs import optimize_affs, get_cell_bindings
from .common import getSetup

path_here = Path(__file__).parent.parent


def makeFigure():
    ax, f = getSetup((15, 5), (1, 2))

    # Parameters
    targCell = "Treg"
    sample_size = 500
    cell_categorization = "CellType2"
    model_valencies = np.array([[(2), (2)]])
    dose = 10e-2
    signal_receptor = "CD122"
    target_receptors = ["CD25", "CD4-1", "CD27", "TIGIT"]
    # target_receptors = ["CD25", "CD4-1"]
    affinity_bounds = (6, 14)
    num_conv_factors = 15

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
        col for col in filterDF.columns if col not in ["Cell Type"] and col != signal_receptor
    ]

    on_target_mask = (filterDF["Cell Type"] == targCell).to_numpy()
    off_target_mask = ~on_target_mask

    # Calculate raw selectivity baseline
    raw_targ_bound_sigR = []
    raw_off_targ_bound_sigR = []
    raw_targ_bound_tarR = []
    raw_off_targ_bound_tarR = []
    for receptor in receptors_of_interest:
        targRecs = filterDF[[signal_receptor, receptor]].to_numpy()
        offTargRecs = filterDF[[signal_receptor, receptor]].to_numpy()
        _, optAffs = optimize_affs(
            targRecs=targRecs[on_target_mask],
            offTargRecs=offTargRecs[off_target_mask],
            dose=dose,
            valencies=model_valencies,
            bounds=affinity_bounds,
        )
        targ_bound = get_cell_bindings(
            recCounts=targRecs[on_target_mask],
            monomerAffs=optAffs,
            dose=dose,
            valencies=model_valencies,
        )
        off_targ_bound = get_cell_bindings(
            recCounts=offTargRecs[off_target_mask],
            monomerAffs=optAffs,
            dose=dose,
            valencies=model_valencies,
        )

        targ_bound_mean = np.sum(targ_bound, axis=0) / targ_bound.shape[0]
        off_targ_bound_mean = np.sum(off_targ_bound, axis=0) / off_targ_bound.shape[0]

        raw_targ_bound_sigR.append(targ_bound_mean[0])
        raw_off_targ_bound_sigR.append(off_targ_bound_mean[0])
        raw_targ_bound_tarR.append(targ_bound_mean[1])
        raw_off_targ_bound_tarR.append(off_targ_bound_mean[1])

    # Calculate selectivity for each conversion factor
    conversion_factors = np.logspace(-2, 8, num=num_conv_factors)
    conv_targ_bound_sigR = []
    conv_off_targ_bound_sigR = []
    conv_targ_bound_tarR = []
    conv_off_targ_bound_tarR = []
    for receptor in receptors_of_interest:
        per_rec_targ_bound_sigR = []
        per_rec_off_targ_bound_sigR = []
        per_rec_targ_bound_tarR = []
        per_rec_off_targ_bound_tarR = []
        for conv_fact in conversion_factors:
            test_DF = filterDF.copy()
            test_DF[receptor] = test_DF[receptor] * conv_fact
            targRecs = test_DF[[signal_receptor, receptor]].to_numpy()
            offTargRecs = test_DF[[signal_receptor, receptor]].to_numpy()
            _, optAffs = optimize_affs(
                targRecs=targRecs[on_target_mask],
                offTargRecs=offTargRecs[off_target_mask],
                dose=dose,
                valencies=model_valencies,
                bounds=affinity_bounds,
            )
            targ_bound = get_cell_bindings(
                recCounts=targRecs[on_target_mask],
                monomerAffs=optAffs,
                dose=dose,
                valencies=model_valencies,
            )
            off_targ_bound = get_cell_bindings(
                recCounts=offTargRecs[off_target_mask],
                monomerAffs=optAffs,
                dose=dose,
                valencies=model_valencies,
            )
            targ_bound = np.sum(targ_bound, axis=0) / targ_bound.shape[0]
            off_targ_bound = np.sum(off_targ_bound, axis=0) / off_targ_bound.shape[0]

            per_rec_targ_bound_sigR.append(targ_bound[0])
            per_rec_off_targ_bound_sigR.append(off_targ_bound[0])
            per_rec_targ_bound_tarR.append(targ_bound[1])
            per_rec_off_targ_bound_tarR.append(off_targ_bound[1])

        per_rec_off_targ_bound_sigR = np.nan_to_num(per_rec_off_targ_bound_sigR, nan=0.0)
        per_rec_off_targ_bound_tarR = np.nan_to_num(per_rec_off_targ_bound_tarR, nan=0.0)

        per_rec_targ_bound_sigR = per_rec_targ_bound_sigR / np.mean(
            per_rec_targ_bound_sigR
        )
        per_rec_off_targ_bound_sigR = per_rec_off_targ_bound_sigR / np.mean(
            per_rec_off_targ_bound_sigR
        )
        per_rec_targ_bound_tarR = per_rec_targ_bound_tarR / np.mean(
            per_rec_targ_bound_tarR
        )
        per_rec_off_targ_bound_tarR = per_rec_off_targ_bound_tarR / np.mean(
            per_rec_off_targ_bound_tarR
        )

        conv_targ_bound_sigR.append(per_rec_targ_bound_sigR)
        conv_off_targ_bound_sigR.append(per_rec_off_targ_bound_sigR)
        conv_targ_bound_tarR.append(per_rec_targ_bound_tarR)
        conv_off_targ_bound_tarR.append(per_rec_off_targ_bound_tarR)

    # Plotting
    palette = sns.color_palette("colorblind", n_colors=len(receptors_of_interest))
    for i, receptor in enumerate(receptors_of_interest):
        ax[0].plot(
            conversion_factors,
            conv_targ_bound_sigR[i],
            marker="o",
            ls="-",
            label=receptor + " (" + signal_receptor + ")",
            color=palette[i],
        )
        ax[0].plot(
            conversion_factors,
            conv_targ_bound_tarR[i],
            marker="*",
            ls="--",
            label=receptor + " (Target)",
            color=palette[i],
        )
        # ax[0].axhline(
        #     y=raw_targ_bound_sigR[i], linestyle="-", color=palette[i], alpha=0.7
        # )
        # ax[0].axhline(
        #     y=raw_targ_bound_tarR[i], linestyle="--", color=palette[i], alpha=0.7
        # )
    ax[0].axvline(x=10**0, linestyle="--", color="black", alpha=0.5)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Conversion Factor")
    ax[0].set_ylabel("Average Bound Receptors")
    ax[0].set_title("Target Cell Bound Receptors vs Conversion Factor")
    ax[0].legend(title="Receptor")

    for i, receptor in enumerate(receptors_of_interest):
        ax[1].plot(
            conversion_factors,
            conv_off_targ_bound_sigR[i],
            marker="o",
            ls="-",
            label=receptor + " (" + signal_receptor + ")",
            color=palette[i],
        )
        ax[1].plot(
            conversion_factors,
            conv_off_targ_bound_tarR[i],
            marker="*",
            ls="--",
            label=receptor + " (Target)",
            color=palette[i],
        )
        # ax[1].axhline(
        #     y=raw_off_targ_bound_sigR[i], linestyle="-", color=palette[i], alpha=0.7
        # )
        # ax[1].axhline(
        #     y=raw_off_targ_bound_tarR[i], linestyle="--", color=palette[i], alpha=0.7
        # )
    ax[1].axvline(x=10**0, linestyle="--", color="black", alpha=0.5)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("Conversion Factor")
    ax[1].set_ylabel("Average Bound Receptors")
    ax[1].set_title("Off-Target Bound Receptors vs Conversion Factor")
    ax[1].legend(title="Receptor")

    return f



    
    