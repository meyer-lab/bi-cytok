"""
Generates a heatmap of optimal selectivities achieved by multivalent complexes
    composed of ligands for various relevant receptors.

Data Import:
- The CITE-seq dataframe (`importCITE`)

Parameters:
- receptors: list of receptors to be analyzed
- cell_type: cell type whose selectivity will be maximized
- dose: dose of ligand to be used in the selectivity calculation
- valency: valency of the complex to be used in the selectivity calculation
- cell_categorization: column name in CITE-seq dataframe for cell type categorization

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
    cell_categorization = "CellType2"

    CITE_DF = importCITE()

    epitopes = [
        col
        for col in CITE_DF.columns
        if col not in ["CellType1", "CellType2", "CellType3"]
    ]
    epitopesDF = CITE_DF[epitopes + [cell_categorization]]
    epitopesDF = epitopesDF.rename(columns={cell_categorization: "Cell Type"})
    sampleDF = sample_receptor_abundances(epitopesDF, 100, cell_type)

    targ_mask = (sampleDF["Cell Type"] == cell_type).to_numpy()
    off_targ_mask = ~targ_mask

    selectivities = np.full((len(receptors), len(receptors)), np.nan)
    for i, rec1 in enumerate(receptors):
        for j, rec2 in enumerate(receptors):
            rec1_abun = sampleDF[[rec1]].to_numpy()
            rec2_abun = sampleDF[[rec2]].to_numpy()

            receptor_abuns = np.hstack((rec1_abun, rec2_abun))

            targ_abun = receptor_abuns[targ_mask]
            off_targ_abun = receptor_abuns[off_targ_mask]

            opt_selec, _ = optimize_affs(
                targ_abun, off_targ_abun, dose, valencies=valency
            )
            selectivities[i, j] = 1 / opt_selec

    selecDF = pd.DataFrame(selectivities, index=receptors, columns=receptors)

    sns.heatmap(
        selecDF, cmap="bwr", annot=True, ax=ax[0], cbar=True, annot_kws={"fontsize": 16}
    )
    ax[0].set_title("Binding model selectivity")

    return f
