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
    ax, f = getSetup((12, 6), (1, 1))

    signal = ["CD122"]
    # receptors = ["CD25", "CD4-1", "CD27", "CD4-2", "CD278"]
    cell_type = "Treg"
    dose = 10e-2
    valency = np.array([[2, 1, 1]])
    cell_categorization = "CellType2"
    sample_size = 100

    CITE_DF = importCITE()

    epitopes = [
        col
        for col in CITE_DF.columns
        if col not in ["CellType1", "CellType2", "CellType3"]
    ]
    receptors = epitopes
    epitopesDF = CITE_DF[epitopes + [cell_categorization]]
    epitopesDF = epitopesDF.rename(columns={cell_categorization: "Cell Type"})
    sampleDF = sample_receptor_abundances(epitopesDF, sample_size, cell_type)

    targ_mask = (sampleDF["Cell Type"] == cell_type).to_numpy()
    off_targ_mask = ~targ_mask

    signal_abun = sampleDF[signal].to_numpy()

    selectivities = np.full((len(receptors), len(receptors)), np.nan)

    row, col = np.tril_indices(len(receptors), k=0)
    for i, j in zip(row, col, strict=False):
        rec1 = receptors[i]
        rec2 = receptors[j]

        rec1_abun = sampleDF[[rec1]].to_numpy()
        rec2_abun = sampleDF[[rec2]].to_numpy()

        receptor_abuns = np.hstack((signal_abun, rec1_abun, rec2_abun))

        targ_abun = receptor_abuns[targ_mask]
        off_targ_abun = receptor_abuns[off_targ_mask]

        opt_selec, _ = optimize_affs(
            targ_abun, off_targ_abun, dose, valencies=valency
        )
        selectivities[i, j] = 1 / opt_selec

    # Symmetrize the matrix by copying values from lower triangle to upper triangle
    i_upper, j_upper = np.triu_indices(len(receptors), k=1)
    selectivities[i_upper, j_upper] = selectivities[j_upper, i_upper]
    selecDF = pd.DataFrame(selectivities, index=receptors, columns=receptors)

    # Remove rows and columns with all NaN values
    selecDF = selecDF.dropna(how='all').dropna(how='all', axis=1)
    selecDF_row_means = selecDF.mean(axis=1)
    selecDF_col_means = selecDF.mean(axis=0)
    selec_thresh = np.percentile(np.concatenate([selecDF_row_means, selecDF_col_means]), 25)
    selecDF = selecDF.loc[
        (selecDF_row_means >= selec_thresh) & (selecDF_col_means >= selec_thresh)
    ]

    sns.heatmap(
        selecDF, cmap="bwr", ax=ax[0], cbar=True, xticklabels=True, yticklabels=True
    )
    ax[0].tick_params(axis='x', labelsize=5)
    ax[0].tick_params(axis='y', labelsize=5)
    ax[0].set_title("Binding model selectivity")

    return f
