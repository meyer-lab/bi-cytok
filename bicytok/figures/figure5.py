"""
Generates line plots to visualize the relationship between
    KL Divergence, Earth Mover's Distance, and Correlation versus Selectivity
    across varying ligand valencies for target and off-target cell types
    using CITE-seq data.

Data Import:
- Loads the CITE-seq dataframe (`importCITE`) and sets up plotting (`getSetup`).
- Defines experimental parameters, including signal receptor (`CD122`), valencies,
    target receptor combinations, target and off-target cell types, and dosage.
- Reads epitope information from a CSV file and samples their abundances
    across target cells using `getSampleAbundances`.

Data Collection:
- Iterates over specified valencies (`[1, 2, 4]`)
    and target receptor combinations (e.g., `["CD25", "CD278"]`).
- For each valency and target receptor combination:
- Optimizes ligand-receptor affinities using `optimizeDesign`.
- Filters the CITE-seq dataframe for relevant marker columns
    corresponding to the target receptors.

Target and Off-Target Cell Definition*:
- Defines binary arrays indicating on-target cells (`Tregs`)
    and off-target cells based on the `offTargState` parameter:
    - `offTargState = 0`: All non-memory Tregs.
    - `offTargState = 1`: All non-Tregs.
    - `offTargState = 2`: Naive Tregs only.

Metric Calculation:
- Computes the following metrics for each marker subset:
    - **KL Divergence** (`KL_divergence_2D`): Measures the divergence
    between on-target and off-target marker distributions.
    - **Earth Mover's Distance** (`EMD_2D`): Quantifies the minimal "effort"
    to transform one distribution into another.
    - **Correlation** (`correlation`): Anti-correlation
    between selected target receptors (measured using CITE-seq data).

Visualization:
- Creates line plots for each metric against selectivity:
    - **KL Divergence vs. Selectivity**: Plotted on a logarithmic scale
    to capture variations in divergence.
    - **EMD vs. Selectivity**: Plotted on a logarithmic scale
    to highlight differences in distribution shifts.
    - **Correlation vs. Selectivity**: Shows the impact
    of receptor anti-correlation on ligand selectivity.
- Uses different hues to indicate valency levels,
    providing a visual comparison across varying ligand valencies.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

from ..distance_metric_funcs import KL_EMD_2D
from ..imports import importCITE
from ..selectivity_funcs import optimize_affs, sample_receptor_abundances
from .common import getSetup

path_here = Path(__file__).parent.parent


def makeFigure():
    ax, f = getSetup((14, 7), (1, 2))

    np.random.seed(98)
    seed = 98

    # Parameters
    sample_size = 100
    targCell = "Treg"
    signal_receptor = "CD122"
    cplx = [(1, 1), (2, 2)]
    allTargets = [
        ["CD25", "CD278"],
        ["CD25", "CD4-1"],
        ["CD25", "CD45RB"],
        ["CD27", "CD45RB"],
        ["CD25", "CD25"],
        ["CD278", "CD28"],
        ["CD4-1", "CD4-2"],
        ["CD27", "CD45RB"],
        ["TIGIT", "CD25"],
        ["CD27", "CD25"],
        ["TCR-2", "CD25"],
        ["CD4-2", "CD25"],
        ["CD122", "CD25"],
        ["TIGIT", "CD122"],
        ["CD27", "CD122"],
        ["TCR-2", "CD122"],
        ["CD4-2", "CD122"],
        ["CD122", "CD122"],
        ["CD25", "CD122"],
        ["CD278", "CD122"],
        ["TIGIT", "TIGIT"],
        ["CD27", "TIGIT"],
        ["TCR-2", "TIGIT"],
        ["CD4-2", "TIGIT"],
        ["CD122", "TIGIT"],
        ["CD25", "TIGIT"],
        ["CD278", "TIGIT"],
        ["TIGIT", "CD278"],
        ["CD27", "CD278"],
        ["TCR-2", "CD278"],
        ["CD4-2", "CD278"],
        ["CD122", "CD278"],
        ["CD278", "CD278"],
    ]
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

    # Imports
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
    KL_div_vals = []
    EMD_vals = []
    Rbound_vals = []
    for targets in allTargets:
        rec_abundances = sampleDF[targets].to_numpy()
        KL_div_mat, EMD_mat = KL_EMD_2D(
            rec_abundances, on_target_mask, off_target_mask, calc_1D=False
        )
        KL_div = KL_div_mat[1, 0]
        EMD = EMD_mat[1, 0]
        KL_div_vals.append(KL_div)
        EMD_vals.append(EMD)

        # Selectivity calculation for each valency
        for valency_pair in cplx:
            modelValencies = np.array([[1] + list(valency_pair)])
            targRecs = dfTargCell[[signal_receptor] + targets].to_numpy()
            offTargRecs = dfOffTargCell[[signal_receptor] + targets].to_numpy()

            optSelec, optAffs = optimize_affs(
                targRecs=targRecs,
                offTargRecs=offTargRecs,
                dose=dose,
                valencies=modelValencies,
            )
            Rbound_vals.append(1 / optSelec)
            


    # Modify valency labels
    valency_map = {"(1, 1)": "Valency 2", "(2, 2)": "Valency 4"}
    valency_labels = [valency_map[str(v)] for _ in allTargets for v in cplx]

    metrics_df = pd.DataFrame(
        {
            "Receptor Pair": [str(pair) for pair in allTargets for _ in cplx],
            "Valency": valency_labels,
            "KL Divergence": np.repeat(KL_div_vals, len(cplx)),
            "EMD": np.repeat(EMD_vals, len(cplx)),
            "Selectivity (Rbound)": Rbound_vals,
            
        }
    )
    # Plot KL vs Selectivity 
    sns.scatterplot(
        data=metrics_df,
        x="KL Divergence",
        y="Selectivity (Rbound)",
        hue="Receptor Pair",
        style="Valency",
        s=70,  # Increase point size
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
        s=70,  # Increase point size
        ax=ax[1],
        legend=True,
    )
    ax[1].legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=True)


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
            f"KL Divergence vs Selectivity\nValency 2 (r = {kl_corr_valency_2:.3f}), ",
            f"Valency 4 (r = {kl_corr_valency_4:.3f}, Seed = {seed})",
        ),
        fontsize=16,
    )
    ax[1].set_title(
        (
            f"EMD vs Selectivity\nValency 2 (r = {emd_corr_valency_2:.3f}), ",
            f"Valency 4 (r = {emd_corr_valency_4:.3f})",
        ),
        fontsize=16,
    )

    
    ax[0].set_xlabel("KL Divergence", fontsize=14)
    ax[0].set_ylabel("Selectivity (Rbound)", fontsize=14)
    ax[1].set_xlabel("EMD", fontsize=14)
    ax[1].set_ylabel("Selectivity (Rbound)", fontsize=14)


    for a in ax:
        a.tick_params(axis="both", labelsize=12)
        sns.despine(ax=a, trim=True)
        for spine in a.spines.values():
            spine.set_linewidth(1.5)
        a.legend(fontsize=12, loc="best")

    return f
