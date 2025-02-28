"""
Generates line plots to visualize the relationship between
    KL Divergence/Earth Mover's Distance and Selectivity across
    varying ligand valencies

Data Import:
- The CITE-seq dataframe (`importCITE`)
- Reads a list of epitopes from a CSV file (`epitopeList.csv`)

Parameters:
- receptors: list of receptors to be analyzed
- signal_receptor: receptor intended to bind to impart effects
- sample_size: number of cells to sample for analysis
    (if greater than available cells, will use all)
- targCell: cell type whose selectivity will be maximized
- test_valencies: list of valencies to be analyzed
- dose: dose of ligand to be used
- cellTypes: array of all relevant cell types

Data Collection:
- Iterates over specified valencies and receptors
- For each valency and receptor, calculates optimal selectivity for a complex
    composed of that receptor's ligand and the signal receptor's ligand whose
    composition is given by the valency
- For each receptor, calculates 2D KL Divergence and Earth Mover's Distance
    between target and off-target cell distributions of that receptor and the
    signal receptor

Outputs:
- Plots KL Divergence and EMD values against Selectivity for each receptor
    and valency combination
- Includes Pearson correlation coefficients for each plot, indicating the
    strength of the linear relationship between the distance metrics and selectivity
"""

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

from ..distance_metric_funcs import KL_EMD_2D
from ..imports import importCITE, sample_receptor_abundances
from ..selectivity_funcs import optimize_affs
from .common import getSetup

path_here = Path(__file__).parent.parent


def makeFigure():
    ax, f = getSetup((14, 7), (1, 2))

    receptors = [
        ["CD25"],
        ["CD278"],
        ["CD4-1"],
        ["CD27"],
        ["CD45RB"],
        ["CD28"],
        ["TCR-2"],
        ["TIGIT"],
    ]
    signal_receptor = "CD122"
    sample_size = 1000
    targCell = "Treg"
    test_valencies = [(1), (2)]
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

    offTargCells = cellTypes[cellTypes != targCell]
    cell_categorization = "CellType2"

    # Load data
    epitopesList = pd.read_csv(path_here / "data" / "epitopeList.csv")
    epitopes = list(epitopesList["Epitope"].unique())
    CITE_DF = importCITE()

    assert targCell in CITE_DF[cell_categorization].unique()

    # Randomly sample a subset of rows
    epitopesDF = CITE_DF[epitopes + [cell_categorization]]
    epitopesDF = epitopesDF.loc[epitopesDF[cell_categorization].isin(cellTypes)]
    epitopesDF = epitopesDF.rename(columns={cell_categorization: "Cell Type"})
    sampleDF = sample_receptor_abundances(
        CITE_DF=epitopesDF,
        numCells=min(sample_size, epitopesDF.shape[0]),
        targCellType=targCell,
        offTargCellTypes=offTargCells,
    )

    # Define target and off-target cell masks (for distance metrics)
    on_target_mask = (sampleDF["Cell Type"] == targCell).to_numpy()
    off_target_mask = sampleDF["Cell Type"].isin(offTargCells).to_numpy()

    # Define target and off-target cell dataframes (for model)
    dfTargCell = sampleDF.loc[on_target_mask]
    dfOffTargCell = sampleDF.loc[off_target_mask]

    selectivity_vals = []
    KL_div_vals = []
    EMD_vals = []
    for receptor in receptors:
        rec_abundances = sampleDF[receptor].to_numpy()

        KL_div_mat, EMD_mat = KL_EMD_2D(
            rec_abundances, on_target_mask, off_target_mask, calc_1D=True
        )
        KL_div = KL_div_mat[0]
        EMD = EMD_mat[0]
        KL_div_vals.append(KL_div)
        EMD_vals.append(EMD)

        # Selectivity calculation for each valency
        for valency in test_valencies:
            model_valencies = np.array([[valency, valency]])
            targRecs = dfTargCell[[signal_receptor] + receptor].to_numpy()
            offTargRecs = dfOffTargCell[[signal_receptor] + receptor].to_numpy()
            optSelec, _ = optimize_affs(
                targRecs=targRecs,
                offTargRecs=offTargRecs,
                dose=dose,
                valencies=model_valencies,
            )
            selectivity_vals.append(1 / optSelec)

    valency_map = {1: "Valency 2", 2: "Valency 4"}
    valency_labels = [valency_map[v] for _ in receptors for v in test_valencies]
    metrics_df = pd.DataFrame(
        {
            "Receptor Pair": [
                str(receptor) for receptor in receptors for _ in test_valencies
            ],
            "Valency": valency_labels,
            "KL Divergence": np.repeat(KL_div_vals, len(test_valencies)),
            "EMD": np.repeat(EMD_vals, len(test_valencies)),
            "Selectivity (Rbound)": selectivity_vals,
        }
    )

    # Plot KL vs Selectivity
    sns.scatterplot(
        data=metrics_df,
        x="KL Divergence",
        y="Selectivity (Rbound)",
        hue="Receptor Pair",
        style="Valency",
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
        style="Valency",
        s=70,
        ax=ax[1],
        legend=True,
    )
    ax[1].legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=True)

    # Calculate Pearson correlations
    valency_2_df = metrics_df[metrics_df["Valency"] == "Valency 2"]
    valency_4_df = metrics_df[metrics_df["Valency"] == "Valency 4"]
    valency_2_df = valency_2_df.dropna(subset=["KL Divergence", "Selectivity (Rbound)"])
    valency_4_df = valency_4_df.dropna(subset=["KL Divergence", "Selectivity (Rbound)"])
    valency_2_df = valency_2_df[
        ~np.isinf(valency_2_df[["KL Divergence", "Selectivity (Rbound)"]].values).any(
            axis=1
        )
    ]
    valency_4_df = valency_4_df[
        ~np.isinf(valency_4_df[["KL Divergence", "Selectivity (Rbound)"]].values).any(
            axis=1
        )
    ]

    # Calculate Pearson correlations for Valency 2
    kl_corr_valency_2, _ = pearsonr(
        valency_2_df["KL Divergence"], valency_2_df["Selectivity (Rbound)"]
    )
    emd_corr_valency_2, _ = pearsonr(
        valency_2_df["EMD"], valency_2_df["Selectivity (Rbound)"]
    )

    # Calculate Pearson correlations for Valency 4
    kl_corr_valency_4, _ = pearsonr(
        valency_4_df["KL Divergence"], valency_4_df["Selectivity (Rbound)"]
    )
    emd_corr_valency_4, _ = pearsonr(
        valency_4_df["EMD"], valency_4_df["Selectivity (Rbound)"]
    )
    ax[0].set_title(
        (
            f"KL Divergence vs Selectivity, Valency 2 (r = {kl_corr_valency_2:.3f}), "
            f"Valency 4 (r = {kl_corr_valency_4:.3f})",
        ),
        fontsize=13,
    )
    ax[1].set_title(
        (
            f"EMD vs Selectivity, Valency 2 (r = {emd_corr_valency_2:.3f}), ",
            f"Valency 4 (r = {emd_corr_valency_4:.3f})",
        ),
        fontsize=13,
    )
    return f
