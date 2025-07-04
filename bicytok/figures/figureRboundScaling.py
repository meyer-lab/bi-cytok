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
    ax, f = getSetup((18, 5), (2, 3))  # Increased width to accommodate 6 subplots

    # Parameters
    targCell = "Treg"
    sample_size = 100
    cell_categorization = "CellType2"
    model_valencies = np.array([[(2), (2)]])
    dose = 10e-2
    signal_receptor = "CD122"
    # target_receptors = ["CD25", "CD4-1", "CD27", "TIGIT"]
    # target_receptors = ["CD25", "TIGIT"]
    # target_receptors = ["CD25", "CD4-1", "TIGIT"]
    target_receptors = ["CD25"]
    affinity_bounds = (-10, 25)
    num_conv_factors = 100

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

    # signal_mean = np.mean(sampleDF[signal_receptor])
    # for receptor in target_receptors:
    #     target_mean = np.mean(sampleDF[receptor])
    #     sampleDF[receptor] = sampleDF[receptor] * (signal_mean / target_mean)

    filterDF = pd.DataFrame(
        {
            signal_receptor: sampleDF[signal_receptor],
            "Cell Type": sampleDF["Cell Type"],
        }
    )
    for receptor in target_receptors:
        filterDF[receptor] = sampleDF[receptor]

    print(np.mean(filterDF[signal_receptor]))

    receptors_of_interest = [
        col for col in filterDF.columns if col not in ["Cell Type"] and col != signal_receptor
    ]

    on_target_mask = (filterDF["Cell Type"] == targCell).to_numpy()
    off_target_mask = ~on_target_mask

    # # Calculate raw selectivity baseline
    # raw_targ_bound_sigR = []
    # raw_off_targ_bound_sigR = []
    # raw_targ_bound_tarR = []
    # raw_off_targ_bound_tarR = []
    # for receptor in receptors_of_interest:
    #     targRecs = filterDF[[signal_receptor, receptor]].to_numpy()
    #     offTargRecs = filterDF[[signal_receptor, receptor]].to_numpy()
    #     _, optAffs = optimize_affs(
    #         targRecs=targRecs[on_target_mask],
    #         offTargRecs=offTargRecs[off_target_mask],
    #         dose=dose,
    #         valencies=model_valencies,
    #         bounds=affinity_bounds,
    #     )
    #     targ_bound = get_cell_bindings(
    #         recCounts=targRecs[on_target_mask],
    #         monomerAffs=optAffs,
    #         dose=dose,
    #         valencies=model_valencies,
    #     )
    #     off_targ_bound = get_cell_bindings(
    #         recCounts=offTargRecs[off_target_mask],
    #         monomerAffs=optAffs,
    #         dose=dose,
    #         valencies=model_valencies,
    #     )

    #     targ_bound_mean = np.sum(targ_bound, axis=0) / targ_bound.shape[0]
    #     off_targ_bound_mean = np.sum(off_targ_bound, axis=0) / off_targ_bound.shape[0]

    #     raw_targ_bound_sigR.append(targ_bound_mean[0])
    #     raw_off_targ_bound_sigR.append(off_targ_bound_mean[0])
    #     raw_targ_bound_tarR.append(targ_bound_mean[1])
    #     raw_off_targ_bound_tarR.append(off_targ_bound_mean[1])

    # Calculate selectivity for each conversion factor
    conversion_factors = np.logspace(-2, 8, num=num_conv_factors)
    conv_targ_bound_sigR = []
    conv_off_targ_bound_sigR = []
    conv_targ_bound_tarR = []
    conv_off_targ_bound_tarR = []
    optimal_selectivities = []
    optimal_target_affinities = []  # Store target receptor affinities
    optimal_signal_affinities = []  # Store signal receptor affinities
    target_mean_losses = []  # Store mean losses for target cells
    off_target_mean_losses = []  # Store mean losses for off-target cells
    convergence_issues = []  # Store conversion factors where optimization failed
    
    for receptor in receptors_of_interest:
        per_rec_targ_bound_sigR = []
        per_rec_off_targ_bound_sigR = []
        per_rec_targ_bound_tarR = []
        per_rec_off_targ_bound_tarR = []
        per_rec_selectivities = []
        per_rec_target_affinities = []  # Store target receptor affinities for this receptor
        per_rec_signal_affinities = []  # Store signal receptor affinities for this receptor
        per_rec_targ_losses = []  # Store target cell losses for this receptor
        per_rec_off_targ_losses = []  # Store off-target cell losses for this receptor
        per_rec_convergence_issues = []  # Store convergence factors with issues for this receptor
        
        for conv_fact in conversion_factors:
            test_DF = filterDF.copy()

            # print(f"Unconverted mean: {np.mean(test_DF[receptor])}, {np.mean(test_DF[signal_receptor])}")

            test_DF[receptor] = test_DF[receptor] * conv_fact
            # test_DF[signal_receptor] = test_DF[signal_receptor] * conv_fact
            rec_mat = test_DF[[signal_receptor, receptor]].to_numpy()

            optSelec, optAffs, converged = optimize_affs(
                targRecs=rec_mat[on_target_mask],
                offTargRecs=rec_mat[off_target_mask],
                dose=dose,
                valencies=model_valencies,
                bounds=affinity_bounds,
            )
            
            # Track convergence issues
            if not converged:
                per_rec_convergence_issues.append(conv_fact)
            
            per_rec_selectivities.append(1 / optSelec)
            per_rec_signal_affinities.append(optAffs[0])  # Signal receptor affinity
            per_rec_target_affinities.append(optAffs[1])  # Target receptor affinity
            
            Rbound, losses = get_cell_bindings(
                recCounts=rec_mat,
                monomerAffs=optAffs,
                dose=dose,
                valencies=model_valencies,
            )
            
            # Calculate mean losses for target and off-target cells
            targ_losses = losses[on_target_mask]
            off_targ_losses = losses[off_target_mask]
            per_rec_targ_losses.append(np.max(targ_losses))
            per_rec_off_targ_losses.append(np.max(off_targ_losses))
            
            targ_bound = Rbound[on_target_mask]
            off_targ_bound = Rbound[off_target_mask]

            targ_bound = np.sum(targ_bound, axis=0) / targ_bound.shape[0]
            off_targ_bound = np.sum(off_targ_bound, axis=0) / off_targ_bound.shape[0]

            print(f"Conversion factor: {conv_fact}, {receptor}: {targ_bound[0] / off_targ_bound[0]}, Converged: {converged}")

            per_rec_targ_bound_sigR.append(targ_bound[0])
            per_rec_off_targ_bound_sigR.append(off_targ_bound[0])
            per_rec_targ_bound_tarR.append(targ_bound[1])
            per_rec_off_targ_bound_tarR.append(off_targ_bound[1])

        conv_targ_bound_sigR.append(per_rec_targ_bound_sigR)
        conv_off_targ_bound_sigR.append(per_rec_off_targ_bound_sigR)
        conv_targ_bound_tarR.append(per_rec_targ_bound_tarR)
        conv_off_targ_bound_tarR.append(per_rec_off_targ_bound_tarR)
        optimal_selectivities.append(per_rec_selectivities)
        optimal_target_affinities.append(per_rec_target_affinities)
        optimal_signal_affinities.append(per_rec_signal_affinities)
        target_mean_losses.append(per_rec_targ_losses)
        off_target_mean_losses.append(per_rec_off_targ_losses)
        convergence_issues.append(per_rec_convergence_issues)

    # Plotting
    palette = sns.color_palette("colorblind", n_colors=len(receptors_of_interest))
    
    # Plot 1: Target receptor bindings across conversion factors
    for i, receptor in enumerate(receptors_of_interest):
        ax[0].plot(
            conversion_factors,
            conv_targ_bound_tarR[i],
            marker="o",
            ls="-",
            color=palette[i],
        )
        ax[0].plot(
            conversion_factors,
            conv_off_targ_bound_tarR[i],
            marker="x",
            ls="--",
            color=palette[i],
        )
        # Add vertical red lines for convergence issues
        for conv_issue in convergence_issues[i]:
            ax[0].axvline(x=conv_issue, color="red", alpha=0.7, linewidth=1)
    ax[0].axvline(x=10**0, linestyle="--", color="black", alpha=0.5)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("Conversion Factor")
    ax[0].set_ylabel("Average Bound Target Receptors")
    ax[0].set_title("Target Receptor Binding vs Conversion Factor")
    # ax[0].legend(title="Cell Type: Target Receptor")

    # Plot 2: Signal receptor bindings across conversion factors
    for i, receptor in enumerate(receptors_of_interest):
        ax[1].plot(
            conversion_factors,
            conv_targ_bound_sigR[i],
            marker="o",
            ls="-",
            label=f"{targCell}: {receptor}",
            color=palette[i],
        )
        ax[1].plot(
            conversion_factors,
            conv_off_targ_bound_sigR[i],
            marker="x",
            ls="--",
            label=f"Non-{targCell}: {receptor}",
            color=palette[i],
        )
        # Add vertical red lines for convergence issues
        for conv_issue in convergence_issues[i]:
            ax[1].axvline(x=conv_issue, color="red", alpha=0.7, linewidth=1)
    ax[1].axvline(x=10**0, linestyle="--", color="black", alpha=0.5)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("Conversion Factor")
    ax[1].set_ylabel("Average Bound Signal Receptors")
    ax[1].set_title(f"Signal Receptor ({signal_receptor}) Binding vs Conversion Factor")
    ax[1].legend(title="Cell Type: Target Receptor")

    # Plot 3: Selectivities across conversion factors
    for i, receptor in enumerate(receptors_of_interest):
        ax[2].plot(
            conversion_factors,
            optimal_selectivities[i],
            marker="o",
            ls="-",
            label=receptor,
            color=palette[i],
        )
        # Add vertical red lines for convergence issues
        for conv_issue in convergence_issues[i]:
            ax[2].axvline(x=conv_issue, color="red", alpha=0.7, linewidth=1)
    ax[2].axvline(x=10**0, linestyle="--", color="black", alpha=0.5)
    ax[2].set_xscale("log")
    ax[2].set_xlabel("Conversion Factor")
    ax[2].set_ylabel("Optimal Selectivity")
    ax[2].set_title("Selectivity vs Conversion Factor")
    ax[2].legend(title="Target Receptor")

    # Plot 4: Target receptor optimal affinities
    for i, receptor in enumerate(receptors_of_interest):
        ax[3].plot(
            conversion_factors,
            optimal_target_affinities[i],
            marker="o",
            ls="-",
            label=receptor,
            color=palette[i],
        )
        # Add vertical red lines for convergence issues
        for conv_issue in convergence_issues[i]:
            ax[3].axvline(x=conv_issue, color="red", alpha=0.7, linewidth=1)
    ax[3].axvline(x=10**0, linestyle="--", color="black", alpha=0.5)
    ax[3].set_xscale("log")
    ax[3].set_xlabel("Conversion Factor")
    ax[3].set_ylabel("Optimal Affinity (log10 Ka)")
    ax[3].set_title(f"Target Receptor Optimal Affinity vs Conversion Factor")
    ax[3].legend(title="Target Receptor")

    # Plot 5: Signal receptor optimal affinities
    for i, receptor in enumerate(receptors_of_interest):
        ax[4].plot(
            conversion_factors,
            optimal_signal_affinities[i],
            marker="o",
            ls="-",
            label=receptor,
            color=palette[i],
        )
        # Add vertical red lines for convergence issues
        for conv_issue in convergence_issues[i]:
            ax[4].axvline(x=conv_issue, color="red", alpha=0.7, linewidth=1)
    ax[4].axvline(x=10**0, linestyle="--", color="black", alpha=0.5)
    ax[4].set_xscale("log")
    ax[4].set_xlabel("Conversion Factor")
    ax[4].set_ylabel("Optimal Affinity (log10 Ka)")
    ax[4].set_title(f"Signal Receptor ({signal_receptor}) Optimal Affinity vs Conversion Factor")
    ax[4].legend(title="Target Receptor")

    # Plot 6: Binding model losses
    for i, receptor in enumerate(receptors_of_interest):
        ax[5].plot(
            conversion_factors,
            target_mean_losses[i],
            marker="o",
            ls="-",
            label=f"{targCell}: {receptor}",
            color=palette[i],
        )
        ax[5].plot(
            conversion_factors,
            off_target_mean_losses[i],
            marker="x",
            ls="--",
            label=f"Non-{targCell}: {receptor}",
            color=palette[i],
        )
        # Add vertical red lines for convergence issues
        for conv_issue in convergence_issues[i]:
            ax[5].axvline(x=conv_issue, color="red", alpha=0.7, linewidth=1)
    ax[5].axvline(x=10**0, linestyle="--", color="black", alpha=0.5)
    ax[5].set_xscale("log")
    ax[5].set_yscale("log")
    ax[5].set_xlabel("Conversion Factor")
    ax[5].set_ylabel("Mean Binding Model Loss")
    ax[5].set_title("Binding Model Loss vs Conversion Factor")
    ax[5].legend(title="Cell Type: Target Receptor")

    return f




