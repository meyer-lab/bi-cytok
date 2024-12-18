from os.path import dirname, join

import numpy as np
import pandas as pd
import seaborn as sns

from ..distanceMetricFuncs import KL_EMD_2D
from ..imports import importCITE
from ..selectivityFuncs import getSampleAbundances, optimizeDesign
from .common import getSetup

path_here = dirname(dirname(__file__))


def makeFigure():
    """
    Generates line plots to visualize the relationship between KL Divergence, Earth Mover's Distance, and Correlation versus Selectivity across varying ligand valencies for target and off-target cell types using CITE-seq data.

    Data Import:
    - Loads the CITE-seq dataframe (`importCITE`) and sets up plotting (`getSetup`).
    - Defines experimental parameters, including signal receptor (`CD122`), valencies, target receptor combinations,
     target and off-target cell types, and dosage.
    - Reads epitope information from a CSV file and samples their abundances across target cells using `getSampleAbundances`.

    Data Collection:
    - Iterates over specified valencies (`[1, 2, 4]`) and target receptor combinations (e.g., `["CD25", "CD278"]`).
    - For each valency and target receptor combination:
    - Optimizes ligand-receptor affinities using `optimizeDesign`.
    - Filters the CITE-seq dataframe for relevant marker columns corresponding to the target receptors.

    Target and Off-Target Cell Definition*:
     Defines binary arrays indicating on-target cells (`Tregs`) and off-target cells based on the `offTargState` parameter:
     - `offTargState = 0`: All non-memory Tregs.
     - `offTargState = 1`: All non-Tregs.
     - `offTargState = 2`: Naive Tregs only.

    Metric Calculation:
    - Computes the following metrics for each marker subset:
     - **KL Divergence** (`KL_divergence_2D`): Measures the divergence between on-target and off-target marker distributions.
     - **Earth Mover's Distance** (`EMD_2D`): Quantifies the minimal "effort" to transform one distribution into another.
     - **Correlation** (`correlation`): Anti-correlation between selected target receptors (measured using CITE-seq data).

    Visualization:
    - Creates line plots for each metric against selectivity:
     - **KL Divergence vs. Selectivity**: Plotted on a logarithmic scale to capture variations in divergence.
     - **EMD vs. Selectivity**: Plotted on a logarithmic scale to highlight differences in distribution shifts.
     - **Correlation vs. Selectivity**: Shows the impact of receptor anti-correlation on ligand selectivity.
    - Uses different hues to indicate valency levels, providing a visual comparison across varying ligand valencies.



    """
    ax, f = getSetup((9, 3), (1, 3))

    CITE_DF = importCITE()

    signal_receptor = "CD122"
    signal_valency = 1
    valencies = [1, 2, 4]
    allTargets = [["CD25", "CD278"], ["CD25", "CD4-2"], ["CD25", "CD45RB"]]
    dose = 10e-2
    offTargState = 0  # Adjust as needed
    cells = np.array(
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
    targCell = "Treg"
    offTCells = cells[cells != targCell]

    epitopesList = pd.read_csv(join(path_here, "data/epitopeList.csv"))
    epitopes = list(epitopesList["Epitope"].unique())
    epitopesDF = getSampleAbundances(epitopes, cells, numCells=1000)

    df = pd.DataFrame(
        columns=[
            "KL Divergence",
            "Earth Mover's Distance",
            "Correlation",
            "Selectivity",
            "Valency",
        ]
    )

    for val in valencies:
        prevOptAffs = [8.0, 8.0, 8.0]
        for targets in allTargets:
            vals = np.array([[signal_valency, val, val]])

            optParams = optimizeDesign(
                signal_receptor,
                targets,
                targCell,
                offTCells,
                epitopesDF,
                dose,
                vals,
                prevOptAffs,
            )
            prevOptAffs = optParams[1]
            select = (1 / optParams[0],)

            non_marker_columns = ["CellType1", "CellType2", "CellType3", "Cell"]
            marker_columns = CITE_DF.columns[~CITE_DF.columns.isin(non_marker_columns)]
            markerDF = CITE_DF.loc[:, marker_columns]

            # Filter to include only columns related to the target receptors
            receptors_of_interest = targets
            filtered_markerDF = markerDF.loc[
                :,
                markerDF.columns.str.contains(
                    "|".join(receptors_of_interest), case=False
                ),
            ]

            # Create binary arrays for on-target and off-target cell types
            on_target = (CITE_DF["CellType3"] == targCell).to_numpy()

            off_target_conditions = {
                0: (CITE_DF["CellType3"] != targCell),  # All non-memory Tregs
                1: (CITE_DF["CellType2"] != "Treg"),  # All non-Tregs
                2: (CITE_DF["CellType3"] == "Treg Naive"),  # Naive Tregs
            }

            if offTargState in off_target_conditions:
                off_target = off_target_conditions[offTargState].to_numpy()
            else:
                raise ValueError("Invalid offTargState value. Must be 0, 1, or 2.")

            rec_abundances = filtered_markerDF.to_numpy()
            KL_div_vals, EMD_vals = KL_EMD_2D(rec_abundances, on_target, off_target)

            # Calculate Pearson correlation inline (no separate function)
            epitopesList = pd.read_csv("./bicytok/data/epitopeList.csv")
            epitopes = list(epitopesList["Epitope"].unique())
            epitopesDF = getSampleAbundances(epitopes, np.array([targCell]))
            epitopesDF = epitopesDF[epitopesDF["CellType2"] == (targCell)]

            corr = epitopesDF[receptors_of_interest].corr(method="pearson")
            sorted_corr = corr.stack().sort_values(ascending=False)
            sorted_corr_df = pd.DataFrame({"Correlation": sorted_corr})

            # Extract correlation value for the relevant receptors (CD25, CD35)
            corr = sorted_corr_df.loc[
                receptors_of_interest[0], receptors_of_interest[1]
            ]["Correlation"]

            data = {
                "KL Divergence": [KL_div_vals],
                "Earth Mover's Distance": [EMD_vals],
                "Correlation": [corr],
                "Selectivity": select,
                "Valency": [val],
            }
            df_temp = pd.DataFrame(
                data,
                columns=[
                    "KL Divergence",
                    "Earth Mover's Distance",
                    "Correlation",
                    "Selectivity",
                    "Valency",
                ],
            )
            df = pd.concat([df, df_temp], ignore_index=True)

    sns.lineplot(data=df, x="KL Divergence", y="Selectivity", hue="Valency", ax=ax[0])
    sns.lineplot(
        data=df, x="Earth Mover's Distance", y="Selectivity", hue="Valency", ax=ax[1]
    )
    sns.lineplot(data=df, x="Correlation", y="Selectivity", hue="Valency", ax=ax[2])
    ax[0].set(xscale="log")
    ax[1].set(xscale="log")

    return f
