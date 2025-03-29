"""
Generates a scatter plot comparing the 3D selectivity of various receptors
    against their 2D KL Divergence and Earth Mover's Distance (EMD) metrics.

Data Import:
- The CITE-seq dataframe (`importCITE`)

Parameters:
- receptor_pairs: list of receptors to be analyzed
- signal_receptor: receptor intended to bind to impart effects
- sample_size: number of cells to sample for analysis
    (if greater than available cells, will use all)
- targCell: cell type whose selectivity will be maximized
- test_valencies: list of valencies to be analyzed
- dose: dose of ligand to be used
- cell_categorization: column name in CITE-seq dataframe that

Data Collection:
- Iterates over specified valencies and receptors
- For each valency and receptor, calculates optimal selectivity for a complex
    composed of that receptor's ligand and the signal receptor's ligand whose
    composition is given by the valency
- For each receptor, calculates 2D KL Divergence and Earth Mover's Distance
    between target and off-target cell distributions of that receptor and the
    signal receptor

Outputs:
- Plots 2D KL Divergence and EMD values against 3D Selectivity for each receptor
    and valency combination
"""

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

from ..distance_metric_funcs import KL_EMD_2D
from ..imports import filter_receptor_abundances, importCITE, sample_receptor_abundances
from ..selectivity_funcs import optimize_affs
from .common import getSetup

path_here = Path(__file__).parent.parent


def makeFigure():
    ax, f = getSetup((12, 6), (1, 2))

    receptor_pairs = [
        ["CD25", "CD25"],
        ["CD25", "CD4-1"],
        ["CD25", "CD4-2"],
        ["CD25", "CD27"],
        ["CD25", "CD278"],
        ["CD25", "CD146"],
        ["CD25", "CD235ab"],
        ["CD25", "CD338"],
        ["CD4-1", "CD4-1"],
        ["CD4-1", "CD4-2"],
        ["CD4-1", "CD27"],
        ["CD4-1", "CD278"],
        ["CD4-1", "CD146"],
        ["CD235ab", "CD146"],
    ]
    signal_receptor = "CD122"
    sample_size = 100
    targCell = "Treg"
    dose = 10e-2
    cell_categorization = "CellType2"

    CITE_DF = importCITE()

    assert targCell in CITE_DF[cell_categorization].unique()

    epitopes = [
        col
        for col in CITE_DF.columns
        if col not in ["CellType1", "CellType2", "CellType3"]
    ]
    epitopesDF = CITE_DF[epitopes + [cell_categorization]]
    epitopesDF = epitopesDF.rename(columns={cell_categorization: "Cell Type"})
    sampleDF = sample_receptor_abundances(
        CITE_DF=epitopesDF,
        numCells=min(sample_size, epitopesDF.shape[0]),
        targCellType=targCell,
    )
    filtered_sampleDF = filter_receptor_abundances(sampleDF, targCell)

    # Define target and off-target cell masks (for distance metrics)
    on_target_mask = (filtered_sampleDF["Cell Type"] == targCell).to_numpy()
    off_target_mask = ~on_target_mask

    # Define target and off-target cell dataframes (for model)
    dfTargCell = filtered_sampleDF.loc[on_target_mask]
    dfOffTargCell = filtered_sampleDF.loc[off_target_mask]

    selectivity_vals = []
    KL_div_vals = []
    EMD_vals = []
    for receptor_pair in receptor_pairs:
        rec_abundances = filtered_sampleDF[receptor_pair].to_numpy()

        # Calculate the KL Divergence and EMD for the current receptor pair
        KL_div_mat, EMD_mat = KL_EMD_2D(
            rec_abundances, on_target_mask, off_target_mask, calc_1D=False
        )
        KL_div = KL_div_mat[1, 0]
        EMD = EMD_mat[1, 0]
        KL_div_vals.append(KL_div)
        EMD_vals.append(EMD)

        # Selectivity calculation for each valency
        model_valencies = np.array([[(2), (1), (1)]])
        targRecs = dfTargCell[[signal_receptor] + receptor_pair].to_numpy()
        offTargRecs = dfOffTargCell[[signal_receptor] + receptor_pair].to_numpy()
        optSelec, _ = optimize_affs(
            targRecs=targRecs,
            offTargRecs=offTargRecs,
            dose=dose,
            valencies=model_valencies,
        )
        selectivity_vals.append(1 / optSelec)

    metrics_df = pd.DataFrame(
        {
            "Receptor Pair": [str(receptor) for receptor in receptor_pairs],
            "KL Divergence": KL_div_vals,
            "EMD": EMD_vals,
            "Selectivity (Rbound)": selectivity_vals,
        }
    )

    # Plot KL vs Selectivity
    sns.scatterplot(
        data=metrics_df,
        x="KL Divergence",
        y="Selectivity (Rbound)",
        hue="Receptor Pair",
        s=70,
        ax=ax[0],
        legend=False,
    )

    # Plot EMD vs Selectivity
    sns.scatterplot(
        data=metrics_df,
        x="EMD",
        y="Selectivity (Rbound)",
        hue="Receptor Pair",
        s=70,
        ax=ax[1],
        legend=True,
    )
    ax[1].legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=True)

    ax[0].set_title(
        ("KL Divergence vs Selectivity"),
        fontsize=13,
    )
    ax[1].set_title(
        ("EMD vs Selectivity"),
        fontsize=13,
    )

    return f
