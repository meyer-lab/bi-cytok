"""
Generates the "1D" selectivities of receptors for targeting a specific cell type over 
    off-target cell types.

Data Import:
- Loads the CITE-seq dataset using `importCITE`

User inputs:
- Receptors to be analyzed
- Target cell type
- Dose of receptor targeting ligand

Selectivity Calculation:
- Uses the multivalent binding model to predict how many receptors will be bound
    by that receptor's ligand on target versus off-target cells.
- This prediction is made based on the measured receptor abundances in the CITE-seq data.
- The optimal selectivity of each receptor is determined by optimizing the affinity
    between each receptor-ligand pair such that the ratio of off-target to 
    on-target binding is minimized.
- The selectivity measurement represents the optimal ratio of target to off-target 
    binding based on differences in receptor abundances across cell types.

Visualization:
- Displays the optimal selectivities of all receptors in a bar plot.
"""

from pathlib import Path

import numpy as np

from ..imports import importCITE
from ..selectivity_funcs import optimize_affs, sample_receptor_abundances
from .common import getSetup

path_here = Path(__file__).parent.parent


def makeFigure():
    ax, f = getSetup((8, 8), (1, 1))

    receptors = ["CD25", "CD4-1", "CD27", "CD4-2", "CD278"]
    cell_type = "Treg"
    dose = 10e-2

    CITE_DF = importCITE()

    filt_cols = ["CellType1", "CellType3", "Cell"]
    marker_columns = CITE_DF.columns[~CITE_DF.columns.isin(filt_cols)]
    markerDF = CITE_DF.loc[:, marker_columns]
    markerDF = markerDF.rename(columns={"CellType2": "Cell Type"})

    sampleDF = sample_receptor_abundances(markerDF, 100)

    targ_mask = (sampleDF["Cell Type"] == cell_type).to_numpy()
    off_targ_mask = (sampleDF["Cell Type"] != cell_type).to_numpy()

    selectivities = []
    for receptor in receptors:
        receptor_abun = sampleDF[[receptor]].to_numpy()

        targ_abun = receptor_abun[targ_mask]
        off_targ_abun = receptor_abun[off_targ_mask]

        opt_selec, opt_affs = optimize_affs(
            targ_abun, off_targ_abun, dose, valencies=np.array([[1]])
        )
        selectivities.append(1 / opt_selec)

    sort_indices = np.argsort(selectivities)

    ax[0].barh(
        np.array(receptors)[sort_indices],
        np.array(selectivities)[sort_indices],
        color="b",
    )
    ax[0].set_xlabel("Selectivity")
    ax[0].set_title("Selectivity of Receptors in Treg Cells")
    ax[0].invert_yaxis()

    return f
