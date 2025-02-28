"""
Generates a heatmap of optimal selectivities achieved by multivalent complexes
    composed of ligands for various relevant receptors.

Data Import:
- The CITE-seq dataframe (`importCITE`)
- Reads a list of epitopes from a CSV file (`epitopeList.csv`)

Parameters:
- receptors: list of receptors to be analyzed
- cell_type: cell type whose selectivity will be maximized
- dose: dose of ligand to be used in the selectivity calculation
- cellTypes: Array of all relevant cell types
- valency: valency of the complex to be used in the selectivity calculation

Outputs:
- Displays the optimal selectivities of all relevant receptor pairs in a heatmap
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
    ax, f = getSetup((8, 8), (1, 1))

    receptors = ["CD25", "CD4-1", "CD27", "CD4-2", "CD278"]
    cell_type = "Treg"
    dose = 10e-2
    valency = np.array([[2, 2]])
    cellTypes = np.array(
        [
            "CD8 Naive",
            "NK",
            "CD8 TEM",
            "CD4 Naive",
            "CD4 CTL",
            "CD8 TCM",
            "CD8 Proliferating",
            "Treg",
        ]
    )

    offTargCells = cellTypes[cellTypes != cell_type]

    CITE_DF = importCITE()

    filt_cols = ["CellType1", "CellType3", "Cell"]
    marker_columns = CITE_DF.columns[~CITE_DF.columns.isin(filt_cols)]
    markerDF = CITE_DF.loc[:, marker_columns]
    markerDF = markerDF.rename(columns={"CellType2": "Cell Type"})

    sampleDF = sample_receptor_abundances(markerDF, 50, cell_type, offTargCells)

    targ_mask = (sampleDF["Cell Type"] == cell_type).to_numpy()
    off_targ_mask = sampleDF["Cell Type"].isin(offTargCells).to_numpy()

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
