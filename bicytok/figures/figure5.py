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
from scipy.stats import linregress
from ..selectivity_funcs import optimize_affs, sample_receptor_abundances
from .common import getSetup

path_here = Path(__file__).parent.parent


def makeFigure():
    ax, f = getSetup((14, 7), (1, 2))
    np.random.seed(42)

    # Distance metric parameters
    offTargState = 1
    targCell = "Treg"
    sample_size = 1000

    # Binding model parameters
    signal_receptor = "CD122"
    cplx = [(1, 1), (2, 2)]
    allTargets = [["CD25", "CD278"], ["CD25", "CD4-1"], ["CD25", "CD45RB"],
        ["CD27", "CD45RB"],
        ["CD25", "CD25"],
        ["CD278", "CD28"],
        ["CD4-1", "CD4-2"],
        ["CD27", "CD45RB"],]
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

    assert isinstance(offTargState, int)
    assert any(np.array([0, 1, 2]) == offTargState)
    assert not (targCell == "Treg Naive" and offTargState == 2)

    # Imports
    epitopesList = pd.read_csv(path_here / "data" / "epitopeList.csv")
    epitopes = list(epitopesList["Epitope"].unique())
    CITE_DF = importCITE()

    assert targCell in CITE_DF["CellType2"].unique()

    non_marker_columns = ["CellType1", "CellType2", "CellType3", "Cell"]
    marker_columns = CITE_DF.columns[~CITE_DF.columns.isin(non_marker_columns)]
    markerDF = CITE_DF.loc[:, marker_columns]

    on_target = (CITE_DF["CellType2"] == targCell).to_numpy()
    off_target_conditions = {
        0: (CITE_DF["CellType3"] != targCell),  # All non-target cells
        1: (
            (CITE_DF["CellType2"] != "Treg") & (CITE_DF["CellType2"] != targCell)
        ),  # All non-Tregs and non-target cells
        2: (CITE_DF["CellType3"] == "Treg Naive"),  # Naive Tregs
    }
    off_target = off_target_conditions[offTargState].to_numpy()

    # Randomly sample a subset of rows
    subset_indices = np.random.choice(
        len(on_target), size=min(sample_size, len(on_target)), replace=False
    )
    on_target = on_target[subset_indices]
    off_target = off_target[subset_indices]

    KL_div_vals = []
    EMD_vals = []
    Rbound_vals = []

    for targets in allTargets:
        markerDF = CITE_DF.loc[:, targets]
        rec_abundances = markerDF.to_numpy()
        rec_abundances = rec_abundances[subset_indices]

        # Calculate metrics (KL and EMD) once for each receptor pair
        KL_div_mat, EMD_mat = KL_EMD_2D(
            rec_abundances, on_target, off_target, calc_1D=False
        )
        KL_div = KL_div_mat[1, 0]
        EMD = EMD_mat[1, 0]

        KL_div_vals.append(KL_div)
        EMD_vals.append(EMD)

        # Sample receptor abundances
        epitopesDF = CITE_DF[epitopes + ["CellType2"]]
        epitopesDF = epitopesDF.loc[epitopesDF["CellType2"].isin(cellTypes)]
        epitopesDF = epitopesDF.rename(columns={"CellType2": "Cell Type"})
        sampleDF = sample_receptor_abundances(CITE_DF=epitopesDF, numCells=100)

        # Selectivity calculation for each valency
        for valency_pair in cplx:
            modelValencies = np.array([[1] + list(valency_pair)])
            dfTargCell = sampleDF.loc[sampleDF["Cell Type"] == targCell]
            targRecs = dfTargCell[[signal_receptor] + targets]
            dfOffTargCell = sampleDF.loc[sampleDF["Cell Type"].isin(offTargCells)]
            offTargRecs = dfOffTargCell[[signal_receptor] + targets]

            optSelec, _ = optimize_affs(
                targRecs=targRecs.to_numpy(),
                offTargRecs=offTargRecs.to_numpy(),
                dose=dose,
                valencies=modelValencies,
            )
            Rbound_vals.append(1 / optSelec)

    # Modify valency labels
    valency_map = {"(1, 1)": "Valency 2", "(2, 2)": "Valency 4"}
    valency_labels = [
        valency_map[str(v)] for _ in allTargets for v in cplx
    ]

    metrics_df = pd.DataFrame(
        {
            "Receptor Pair": [str(pair) for pair in allTargets for _ in cplx],
            "Valency": valency_labels,
            "KL Divergence": np.repeat(KL_div_vals, len(cplx)),
            "EMD": np.repeat(EMD_vals, len(cplx)),
            "Selectivity (Rbound)": Rbound_vals,
        }
    )
    '''
    def add_best_fit_line(ax, x, y, subset, label):
        slope, intercept, _, _, _ = linregress(x, y)
        ax.plot(x, intercept + slope * x, label=f"{label}: Slope = {slope:.2f}", linestyle="--")
        return slope
    '''
  

    # Plot KL vs Selectivity
    sns.scatterplot(
        data=metrics_df,
        x="KL Divergence",
        y="Selectivity (Rbound)",
        hue="Receptor Pair",
        style="Valency",
        s=70,  # Increase point size
        ax=ax[0],
    )

    # Add best fit lines for Valency 2 and Valency 4 on KL plot
    '''
    valency_2 = metrics_df[metrics_df["Valency"] == "Valency 2"]
    valency_4 = metrics_df[metrics_df["Valency"] == "Valency 4"]
    
    slope_v2 = add_best_fit_line(
        ax[0],
        valency_2["KL Divergence"].values,
        valency_2["Selectivity (Rbound)"].values,
        valency_2,
        "Valency 2",
    )
    slope_v4 = add_best_fit_line(
        ax[0],
        valency_4["KL Divergence"].values,
        valency_4["Selectivity (Rbound)"].values,
        valency_4,
        "Valency 4",
    )
    '''

    # Plot EMD vs Selectivity
    sns.scatterplot(
        data=metrics_df,
        x="EMD",
        y="Selectivity (Rbound)",
        hue="Receptor Pair",
        style="Valency",
        s=70,  # Increase point size
        ax=ax[1],
    )
    '''
    # Add best fit lines for Valency 2 and Valency 4 on EMD plot
    slope_v2_emd = add_best_fit_line(
        ax[1],
        valency_2["EMD"].values,
        valency_2["Selectivity (Rbound)"].values,
        valency_2,
        "Valency 2",
    )
    slope_v4_emd = add_best_fit_line(
        ax[1], 
        valency_4["EMD"].values,
        valency_4["Selectivity (Rbound)"].values,
        valency_4,
        "Valency 4",
    )
    '''
    # Adjust titles, labels, and border

    valency_2_df = metrics_df[metrics_df["Valency"] == "Valency 2"]
    valency_4_df = metrics_df[metrics_df["Valency"] == "Valency 4"]

    # Calculate Pearson correlations for Valency 2
    kl_corr_valency_2, _ = pearsonr(valency_2_df["KL Divergence"], valency_2_df["Selectivity (Rbound)"])
    emd_corr_valency_2, _ = pearsonr(valency_2_df["EMD"], valency_2_df["Selectivity (Rbound)"])

    # Calculate Pearson correlations for Valency 4
    kl_corr_valency_4, _ = pearsonr(valency_4_df["KL Divergence"], valency_4_df["Selectivity (Rbound)"])
    emd_corr_valency_4, _ = pearsonr(valency_4_df["EMD"], valency_4_df["Selectivity (Rbound)"])

    # Update plot titles with both Valency 2 and Valency 4 correlation values
    ax[0].set_title(f"KL Divergence vs Selectivity\nValency 2 (r = {kl_corr_valency_2:.3f}), Valency 4 (r = {kl_corr_valency_4:.3f})", fontsize=16)
    ax[1].set_title(f"EMD vs Selectivity\nValency 2 (r = {emd_corr_valency_2:.3f}), Valency 4 (r = {emd_corr_valency_4:.3f})", fontsize=16)

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
    
