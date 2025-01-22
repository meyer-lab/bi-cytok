"""
Generates the "2D" selectivities of receptors for targeting a specific cell type over
    off-target cell types. The 2D selectivity is different from 1D selectivity in that
    it considers a multivalent ligand complex to target two receptors simultaneously.

Data Import:
- Loads the CITE-seq dataset using `importCITE`

User inputs:
- Receptors to be analyzed
- Target cell type
- Dose of receptor targeting ligand
- Valency of the ligand complex

Selectivity Calculation:
- Uses the multivalent binding model to predict how many receptors will be bound
    by that receptor's ligand on target versus off-target cells.
- This prediction is made based on the measured receptor abundances in the CITE-seq
    data.
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
import pandas as pd
import seaborn as sns

from ..imports import importCITE
from ..selectivity_funcs import optimize_affs, sample_receptor_abundances
from .common import getSetup

path_here = Path(__file__).parent.parent


def makeFigure():
    ax, f = getSetup((8, 8), (1, 1))

    receptors = ["CD25", "CD4-1", "CD27", "CD4-2", "CD278"]
    cell_type = "Treg"
    dose = 10e-2
    valency = np.array([[2, 2]])

    CITE_DF = importCITE()

    filt_cols = ["CellType1", "CellType3", "Cell"]
    marker_columns = CITE_DF.columns[~CITE_DF.columns.isin(filt_cols)]
    markerDF = CITE_DF.loc[:, marker_columns]
    markerDF = markerDF.rename(columns={"CellType2": "Cell Type"})

    sampleDF = sample_receptor_abundances(markerDF, 50)

    targ_mask = (sampleDF["Cell Type"] == cell_type).to_numpy()
    off_targ_mask = (sampleDF["Cell Type"] != cell_type).to_numpy()

    selectivities = np.full((len(receptors), len(receptors)), np.nan)
    for i, rec1 in enumerate(receptors):
        for j, rec2 in enumerate(receptors):
            rec1_abun = sampleDF[[rec1]].to_numpy()
            rec2_abun = sampleDF[[rec2]].to_numpy()

            receptor_abuns = np.hstack((rec1_abun, rec2_abun))

            targ_abun = receptor_abuns[targ_mask]
            off_targ_abun = receptor_abuns[off_targ_mask]

            opt_selec, opt_affs = optimize_affs(
                targ_abun, off_targ_abun, dose, valencies=valency
            )
            selectivities[i, j] = 1 / opt_selec

    selecDF = pd.DataFrame(selectivities, index=receptors, columns=receptors)

    sns.heatmap(
        selecDF, cmap="bwr", annot=True, ax=ax[0], cbar=True, annot_kws={"fontsize": 16}
    )
    ax[0].set_title("Binding model selectivity")

    return f
