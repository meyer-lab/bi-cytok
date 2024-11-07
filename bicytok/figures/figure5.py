from os.path import dirname, join

import numpy as np
import pandas as pd
import seaborn as sns

from ..distanceMetricFuncs import EMD_2D, KL_divergence_2D, correlation
from ..imports import importCITE
from ..selectivityFuncs import getSampleAbundances, optimizeDesign
from .common import getSetup

path_here = dirname(dirname(__file__))


def makeFigure():
    """Figure file to generate plots of bispecific ligand selectivity for combinations of different KL divergences, EMDs, and anti-correlations."""
    ax, f = getSetup((9, 3), (1, 3))

    CITE_DF = importCITE()
    new_df = CITE_DF.sample(1000, random_state=42)

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
            on_target = (CITE_DF["CellType3"] == targCell).astype(int)

            # Define off-target conditions using a dictionary
            off_target_conditions = {
                0: (CITE_DF["CellType3"] != targCell),  # All non-memory Tregs
                1: (CITE_DF["CellType2"] != "Treg"),  # All non-Tregs
                2: (CITE_DF["CellType3"] == "Treg Naive"),  # Naive Tregs
            }

            # Set off_target based on offTargState
            if offTargState in off_target_conditions:
                off_target = off_target_conditions[offTargState].astype(int)
            else:
                raise ValueError("Invalid offTargState value. Must be 0, 1, or 2.")

            # Calculate KL divergence and EMD with filtered data and binary arrays
            KLD = KL_divergence_2D(filtered_markerDF, on_target, off_target)
            EMD = EMD_2D(filtered_markerDF, on_target, off_target)
            corr = correlation(targCell, targets).loc[targets[0], targets[1]][
                "Correlation"
            ]

            data = {
                "KL Divergence": [KLD],
                "Earth Mover's Distance": [EMD],
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
