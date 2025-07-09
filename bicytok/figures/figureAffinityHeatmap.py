"""
Generates contour plots showing the effect of varying signal and target receptor
affinities independently on the selectivity of the signal receptor, across different
conversion factors applied to the target receptor.

Data Import:
- The CITE-seq dataframe (`importCITE`)

Parameters:
- targCell: cell type whose selectivity will be maximized
- sample_size: number of cells to sample from the CITE-seq dataframe
- cell_categorization: column name in the CITE-seq dataframe that categorizes cells
- model_valencies: valencies each receptor's ligand in the model molecule
- dose: dose of the model molecule
- signal_receptor: receptor that is the target of the model molecule
- target_receptor: receptor to be tested for selectivity
- affinity_range: range of affinities to test for both receptors
- num_points: number of points along each affinity axis
- conversion_factors: scaling factors to apply to target receptor counts

Outputs:
- Contour plots showing signal receptor selectivity as a function
  of signal and target receptor affinities
- Separate plots for different conversion factors applied to target receptor
- Selectivity is defined as target cell binding / off-target cell binding for signal
  receptor
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..imports import importCITE, sample_receptor_abundances
from ..selectivity_funcs import get_cell_bindings
from .common import getSetup

path_here = Path(__file__).parent.parent


def makeFigure():
    ax, f = getSetup((18, 6), (1, 3))

    # Parameters
    targCell = "Treg"
    sample_size = 100
    cell_categorization = "CellType2"
    model_valencies = np.array([[(2), (2)]])
    dose = 10e-2
    signal_receptor = "CD122"
    target_receptor = "CD25"
    affinity_range = (-10, 25)
    num_points = 30
    conversion_factors = [1.0, 10.0, 1000.0]

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
            target_receptor: sampleDF[target_receptor],
            "Cell Type": sampleDF["Cell Type"],
        }
    )

    on_target_mask = (filterDF["Cell Type"] == targCell).to_numpy()

    # Create affinity grids
    signal_affs = np.linspace(affinity_range[0], affinity_range[1], num_points)
    target_affs = np.linspace(affinity_range[0], affinity_range[1], num_points)
    Signal_Affs, Target_Affs = np.meshgrid(signal_affs, target_affs)

    rec_mat_original = filterDF[[signal_receptor, target_receptor]].to_numpy()

    # Create contour plots for each conversion factor
    contour_levels = 20

    for plot_idx, conv_factor in enumerate(conversion_factors):
        # Apply conversion factor to target receptor
        rec_mat = rec_mat_original.copy()
        rec_mat[:, 1] = rec_mat[:, 1] * conv_factor  # Scale target receptor

        # Initialize result array for signal receptor selectivity
        signal_selectivity = np.zeros_like(Signal_Affs)

        # Calculate selectivity for each affinity combination
        for i, sig_aff in enumerate(signal_affs):
            for j, targ_aff in enumerate(target_affs):
                affs = np.array([sig_aff, targ_aff])

                Rbound, _ = get_cell_bindings(
                    recCounts=rec_mat,
                    monomerAffs=affs,
                    dose=dose,
                    valencies=model_valencies,
                )

                # Calculate averages for target and off-target cells
                targ_bound = Rbound[on_target_mask]

                # Calculate mean bound signal receptors for each cell type
                targ_bound_signal_mean = np.mean(targ_bound[:, 0])  # Signal receptor
                full_bound_signal_mean = np.mean(Rbound[:, 0])  # Signal receptor

                # Calculate selectivity (target / off-target binding ratio)
                signal_selectivity[j, i] = (
                    targ_bound_signal_mean / full_bound_signal_mean
                )

        # Create contour plot for this conversion factor
        cs = ax[plot_idx].contourf(
            Signal_Affs,
            Target_Affs,
            signal_selectivity,
            levels=contour_levels,
            cmap="viridis",
        )
        ax[plot_idx].set_xlabel(f"{signal_receptor} Affinity (log10 Ka)")
        ax[plot_idx].set_ylabel(f"{target_receptor} Affinity (log10 Ka)")
        ax[plot_idx].set_title(
            f"{signal_receptor} Selectivity\n{target_receptor} {conv_factor:.0f}x"
        )
        cbar = plt.colorbar(cs, ax=ax[plot_idx])
        cbar.set_label("Selectivity Ratio")

    return f
