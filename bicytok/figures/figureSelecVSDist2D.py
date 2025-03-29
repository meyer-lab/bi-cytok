"""
Generates a scatter plot comparing the selectivity of various receptors
    against their KL Divergence and Earth Mover's Distance (EMD) metrics.

Data Import:
- The CITE-seq dataframe (`importCITE`)

Parameters:
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
- Plots KL Divergence and EMD values against Selectivity for each receptor
    and valency combination
"""

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

from ..distance_metric_funcs import KL_EMD_2D
from ..imports import importCITE, sample_receptor_abundances
from ..selectivity_funcs import optimize_affs
from .common import getSetup

path_here = Path(__file__).parent.parent


def makeFigure():
    ax, f = getSetup((14, 7), (1, 2))

    signal_receptor = "CD122"
    sample_size = 5000
    targCell = "Treg"
    test_valencies = [(1), (2)]
    dose = 10e-2
    cell_categorization = "CellType2"

    CITE_DF = importCITE()

    assert targCell in CITE_DF[cell_categorization].unique()

    epitopes = [
        col
        for col in CITE_DF.columns
        if col not in ["Cell", "CellType1", "CellType2", "CellType3"]
    ]
    epitopesDF = CITE_DF[epitopes + [cell_categorization]]
    epitopesDF = epitopesDF.rename(columns={cell_categorization: "Cell Type"})
    sampleDF = sample_receptor_abundances(
        CITE_DF=epitopesDF,
        numCells=min(sample_size, epitopesDF.shape[0]),
        targCellType=targCell,
    )

    # Define target and off-target cell masks (for distance metrics)
    on_target_mask = (sampleDF["Cell Type"] == targCell).to_numpy()
    off_target_mask = ~on_target_mask

    # Define target and off-target cell dataframes (for model)
    dfTargCell = sampleDF.loc[on_target_mask]
    dfOffTargCell = sampleDF.loc[off_target_mask]

    selectivity_vals = []
    KL_div_vals = []
    EMD_vals = []
    for receptor in epitopes:
        rec_abundances = sampleDF[receptor].to_numpy()

        # Calculate the KL Divergence and EMD for the current receptor
        KL_div_mat, EMD_mat = KL_EMD_2D(
            rec_abundances, on_target_mask, off_target_mask, calc_1D=True
        )
        KL_div = KL_div_mat[0]
        EMD = EMD_mat[0]
        KL_div_vals.append(KL_div)
        EMD_vals.append(EMD)

        # Selectivity calculation for each valency
        for valency in test_valencies:
            if np.isnan(KL_div) or np.isnan(EMD):
                selectivity_vals.append(np.nan)
                continue

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
    valency_labels = [valency_map[v] for _ in epitopes for v in test_valencies]
    metrics_df = pd.DataFrame(
        {
            "Receptor Pair": [
                str(receptor) for receptor in epitopes for _ in test_valencies
            ],
            "Valency": valency_labels,
            "KL Divergence": np.repeat(KL_div_vals, len(test_valencies)),
            "EMD": np.repeat(EMD_vals, len(test_valencies)),
            "Selectivity (Rbound)": selectivity_vals,
        }
    )

    # Create a dataframe with unique receptors and their distance metrics
    unique_receptors_df = pd.DataFrame(
        {
            "Receptor Pair": [str(receptor) for receptor in epitopes],
            "KL Divergence": [KL_div_val[0] for KL_div_val in KL_div_vals],
            "EMD": [EMD_val[0] for EMD_val in EMD_vals],
        }
    )
    unique_receptors_df = unique_receptors_df.fillna(0)

    # Get the indices of the top 10 receptors by KL Divergence
    top_kl_indices = unique_receptors_df["KL Divergence"].nlargest(10).index.tolist()
    top_kl_receptors = unique_receptors_df.iloc[top_kl_indices][
        "Receptor Pair"
    ].tolist()

    # Get the indices of the top 10 receptors by EMD
    top_emd_indices = unique_receptors_df["EMD"].nlargest(10).index.tolist()
    top_emd_receptors = unique_receptors_df.iloc[top_emd_indices][
        "Receptor Pair"
    ].tolist()

    # Create filtered dataframes for plotting
    metrics_df_kl_filtered = metrics_df[
        metrics_df["Receptor Pair"].isin(top_kl_receptors)
    ]
    metrics_df_emd_filtered = metrics_df[
        metrics_df["Receptor Pair"].isin(top_emd_receptors)
    ]

    # Plot KL vs Selectivity with only top 10 KL receptors in legend
    sns.scatterplot(
        data=metrics_df,
        x="KL Divergence",
        y="Selectivity (Rbound)",
        hue="Receptor Pair",
        style="Valency",
        s=70,
        ax=ax[0],
        legend=False,
        alpha=0.5,
    )

    # Overlay the top 10 with more visibility
    sns.scatterplot(
        data=metrics_df_kl_filtered,
        x="KL Divergence",
        y="Selectivity (Rbound)",
        hue="Receptor Pair",
        style="Valency",
        s=70,
        ax=ax[0],
        legend=True,
    )
    ax[0].legend(
        loc="upper left", bbox_to_anchor=(1, 1), frameon=True, title="Top 10 by KL Div"
    )

    # Plot EMD vs Selectivity with only top 10 EMD receptors in legend
    sns.scatterplot(
        data=metrics_df,
        x="EMD",
        y="Selectivity (Rbound)",
        hue="Receptor Pair",
        style="Valency",
        s=70,
        ax=ax[1],
        legend=False,
        alpha=0.5,
    )

    # Overlay the top 10 with more visibility
    sns.scatterplot(
        data=metrics_df_emd_filtered,
        x="EMD",
        y="Selectivity (Rbound)",
        hue="Receptor Pair",
        style="Valency",
        s=70,
        ax=ax[1],
        legend=True,
    )
    ax[1].legend(
        loc="upper left", bbox_to_anchor=(1, 1), frameon=True, title="Top 10 by EMD"
    )

    ax[0].set_title(
        ("KL Divergence vs Selectivity"),
        fontsize=13,
    )
    ax[1].set_title(
        ("EMD vs Selectivity"),
        fontsize=13,
    )

    return f
