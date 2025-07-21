"""
Generates a comprehensive analysis of how scaling factors applied to raw receptor counts
    affect multiple aspects of receptor selectivity optimization.

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
- Kx_star_bounds: optimization bounds for Kx_star parameter
- num_conv_factors: number of conversion factors to test (logspace from 0.01 to 1e8)

Outputs:
- A 6-panel figure showing:
  1. Target receptor binding vs conversion factor (on-target vs off-target)
  2. Signal receptor binding vs conversion factor (on-target vs off-target)
  3. Optimal selectivity vs conversion factor
  4. Optimal target receptor affinity vs conversion factor
  5. Optimal signal receptor affinity vs conversion factor
  6. Optimal Kx_star vs conversion factor
- Red vertical lines indicate conversion factors where optimization failed to converge
- Black dashed vertical line at conversion factor = 1 (no scaling reference)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

from ..imports import importCITE, sample_receptor_abundances
from ..selectivity_funcs import get_cell_bindings, optimize_affs
from .common import getSetup

path_here = Path(__file__).parent.parent


def makeFigure():
    ax, f = getSetup((18, 5), (2, 3))  # Increased width to accommodate 6 subplots

    # Parameters
    targCell = "Treg"
    sample_size = 1000
    cell_categorization = "CellType2"
    dose = 10e-2
    signal_receptor = "CD122"
    # receptors_of_interest = [("CD25", 1), ("CD25", 2), ("CD4-1", 1), ("CD4-1", 2)]
    receptors_of_interest = [("CD25", 2), ("CD4-1", 2), ("CD27", 2)]
    affinity_bounds = (6, 12)
    Kx_star_bounds = (2.24e-15, 2.24e-3)
    num_conv_factors = 10
    conversion_factors = np.logspace(-1, 8, num=num_conv_factors)
    # conversion_factors = np.logspace(-5, 2, num=num_conv_factors)
    # conversion_factors = [21544.346900318866]

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
    for receptor_valency in receptors_of_interest:
        receptor = receptor_valency[0]
        filterDF[receptor] = sampleDF[receptor]

    # receptors_of_interest = [
    #     col
    #     for col in filterDF.columns
    #     if col not in ["Cell Type"] and col != signal_receptor
    # ]

    on_target_mask = (filterDF["Cell Type"] == targCell).to_numpy()
    off_target_mask = ~on_target_mask    

    # Structured data storage
    results = {
        "targ_bound_sigR": [],
        "off_targ_bound_sigR": [],
        "targ_bound_tarR": [],
        "off_targ_bound_tarR": [],
        "selectivities": [],
        "target_affinities": [],
        "signal_affinities": [],
        "optimal_Kx_stars": [],
        "convergence_issues": [],
    }

    for receptor_valency in receptors_of_interest:
        # Initialize per-receptor results
        per_rec_results = {key: [] for key in results}

        receptor = receptor_valency[0]
        model_valencies = np.array([[(receptor_valency[1]), (receptor_valency[1])]])

        for conv_fact in conversion_factors:
            test_DF = filterDF.copy()
            test_DF[receptor] = test_DF[receptor] * conv_fact
            # effective_dose = 10**conv_fact
            # test_DF[signal_receptor] = test_DF[signal_receptor] * 332
            rec_mat = test_DF[[signal_receptor, receptor]].to_numpy()

            optSelec, optAffs, optKxStar, converged = optimize_affs(
                targRecs=rec_mat[on_target_mask],
                offTargRecs=rec_mat[off_target_mask],
                dose=dose,
                valencies=model_valencies,
                bounds=affinity_bounds,
                Kx_star_bounds=Kx_star_bounds,
            )

            # Track convergence issues
            if not converged:
                per_rec_results["convergence_issues"].append(conv_fact)

            per_rec_results["selectivities"].append(1 / optSelec)
            per_rec_results["signal_affinities"].append(optAffs[0])
            per_rec_results["target_affinities"].append(optAffs[1])
            per_rec_results["optimal_Kx_stars"].append(optKxStar)

            Rbound, losses = get_cell_bindings(
                recCounts=rec_mat,
                monomerAffs=optAffs,
                dose=dose,
                valencies=model_valencies,
                Kx_star=optKxStar,
            )

            targ_bound = Rbound[on_target_mask]
            off_targ_bound = Rbound[off_target_mask]

            targ_bound = np.sum(targ_bound, axis=0) / targ_bound.shape[0]
            off_targ_bound = np.sum(off_targ_bound, axis=0) / off_targ_bound.shape[0]

            per_rec_results["targ_bound_sigR"].append(targ_bound[0])
            per_rec_results["off_targ_bound_sigR"].append(off_targ_bound[0])
            per_rec_results["targ_bound_tarR"].append(targ_bound[1])
            per_rec_results["off_targ_bound_tarR"].append(off_targ_bound[1])

        # Store results for this receptor
        for key in results:
            results[key].append(per_rec_results[key])

    # Plotting
    palette = sns.color_palette("colorblind", n_colors=len(receptors_of_interest))

    # Define plot configurations
    plot_configs = [
        {
            "data_key": "targ_bound_tarR",
            "off_data_key": "off_targ_bound_tarR",
            "ylabel": "Average Bound Target Receptors",
            "title": "Target Receptor Binding vs Conversion Factor",
        },
        {
            "data_key": "targ_bound_sigR",
            "off_data_key": "off_targ_bound_sigR",
            "ylabel": "Average Bound Signal Receptors",
            "title": f"Signal Receptor ({signal_receptor}) Binding vs Scaling Factor",
        },
        {
            "data_key": "selectivities",
            "ylabel": "Optimal Selectivity",
            "title": "Selectivity vs Conversion Factor",
        },
        {
            "data_key": "target_affinities",
            "ylabel": "Optimal Affinity (log10 Ka)",
            "title": "Target Receptor Optimal Affinity vs Conversion Factor",
        },
        {
            "data_key": "signal_affinities",
            "ylabel": "Optimal Affinity (log10 Ka)",
            "title": f"Signal Receptor ({signal_receptor}) Affinity vs Scaling Factor",
        },
        {
            "data_key": "optimal_Kx_stars",
            "ylabel": "Optimal Kx_star",
            "title": "Optimal Kx_star vs Conversion Factor",
        },
    ]

    for plot_idx, config in enumerate(plot_configs):
        for i, receptor_valency in enumerate(receptors_of_interest):
            receptor = receptor_valency[0]
            valency = receptor_valency[1]
            # Plot main data
            ax[plot_idx].plot(
                conversion_factors,
                results[config["data_key"]][i],
                marker="o",
                ls="-",
                label=f"{targCell}: {receptor} (Valency: {valency*2})"
                if "off_data_key" in config
                else f"{receptor} (Valency: {valency*2})",
                color=palette[i],
            )

            # Plot off-target data if applicable
            if "off_data_key" in config:
                ax[plot_idx].plot(
                    conversion_factors,
                    results[config["off_data_key"]][i],
                    marker="x",
                    ls="--",
                    label=f"Non-{targCell}: {receptor}",
                    color=palette[i],
                )

            # Add vertical red lines for convergence issues
            for conv_issue in results["convergence_issues"][i]:
                ax[plot_idx].axvline(x=conv_issue, color="red", alpha=0.7, linewidth=1)

        # Configure axes
        ax[plot_idx].axvline(x=10**0, linestyle="--", color="black", alpha=0.5)
        ax[plot_idx].set_xscale("log")
        if plot_idx in [0, 1, 2, 5]:  # Log scale for binding, selectivity, and Kx_star plots
            ax[plot_idx].set_yscale("log")
        ax[plot_idx].set_xlabel("Conversion Factor")
        ax[plot_idx].set_ylabel(config["ylabel"])
        ax[plot_idx].set_title(config["title"])

        # Add legend for specific plots
        if plot_idx in [1, 2, 3, 4, 5]:
            legend_title = (
                "Cell Type: Target Receptor"
                if "off_data_key" in config
                else "Target Receptor"
            )
            ax[plot_idx].legend(title=legend_title)

    return f
