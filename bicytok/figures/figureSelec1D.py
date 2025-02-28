"""
Generates a bar plot of optimal selectivities achieved by monovalent complexes of
    ligands for a various set of relevant receptors.

Data Import:
- The CITE-seq dataframe (`importCITE`)

Parameters:
- receptors: list of receptors to be analyzed
- cell_type: cell type whose selectivity will be maximized
- dose: dose of ligand to be used in the selectivity calculation
- cellTypes: Array of all relevant cell types

Outputs:
- Displays the optimal selectivities of all relevant receptors in a bar plot
"""

from pathlib import Path

import numpy as np

from ..imports import importCITE, sample_receptor_abundances
from ..selectivity_funcs import optimize_affs
from .common import getSetup

path_here = Path(__file__).parent.parent


def makeFigure():
    ax, f = getSetup((8, 8), (1, 1))

    receptors = ["CD25", "CD4-1", "CD27", "CD4-2", "CD278"]
    cell_type = "Treg"
    dose = 10e-2
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

    sampleDF = sample_receptor_abundances(markerDF, 100, cell_type, offTargCells)

    targ_mask = (sampleDF["Cell Type"] == cell_type).to_numpy()
    off_targ_mask = sampleDF["Cell Type"].isin(offTargCells).to_numpy()

    selectivities = []
    for receptor in receptors:
        receptor_abun = sampleDF[[receptor]].to_numpy()

        targ_abun = receptor_abun[targ_mask]
        off_targ_abun = receptor_abun[off_targ_mask]

        opt_selec, _ = optimize_affs(
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
