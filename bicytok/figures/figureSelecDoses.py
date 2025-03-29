"""
Figure file to generate dose response curves for any combination
    of multivalent and multispecific ligands.
Outputs dose vs. selectivity for the target cell and amount of target cell
    bound at indicated signal receptor.

Data Import:
- The CITE-seq dataframe (`importCITE`)

Parameters:
- signal: Receptor that the ligand is delivering signal to; selectivity and
    target bound are with respect to engagement with this receptor
- allTargets: List of paired [(target receptor, valency)] combinations for each
    targeting receptor; to be used for targeting the target cell, not signaling
- targCell: cell type whose selectivity will be maximized
- receptors_of_interest: list of receptors to be analyzed

Outputs:
- Plots dose vs. selectivity and dose vs. target bound for each combination
    of targeting receptors
- Each plot is labeled with receptor names on the y-axis and their respective
    values (selectivity or target bound) on the x-axis
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..imports import importCITE, sample_receptor_abundances
from ..selectivity_funcs import (
    get_cell_bindings,
    optimize_affs,
)
from .common import getSetup

path_here = Path(__file__).parent.parent

plt.rcParams["svg.fonttype"] = "none"


def makeFigure():
    ax, f = getSetup((6, 3), (1, 2))

    signal = ["CD122", 1]
    allTargets = [
        [("CD25", 1)],
        [("CD25", 4)],
        [("CD25", 1), ("CD278", 1)],
        [("CD25", 4), ("CD278", 4)],
        [("CD25", 1), ("CD27", 1)],
        [("CD25", 4), ("CD27", 4)],
        [("CD25", 1), ("CD278", 1), ("CD27", 1)],
        [("CD25", 4), ("CD278", 4), ("CD27", 4)],
    ]
    targCell = "Treg"

    CITE_DF = importCITE()

    epitopes = [
        col
        for col in CITE_DF.columns
        if col not in ["CellType1", "CellType2", "CellType3"]
    ]
    epitopesDF = CITE_DF[epitopes + ["CellType2"]]
    epitopesDF = epitopesDF.rename(columns={"CellType2": "Cell Type"})

    sampleDF = sample_receptor_abundances(
        CITE_DF=epitopesDF,
        numCells=100,
        targCellType=targCell,
    )

    doseVec = np.logspace(-2, 2, num=10)
    df = pd.DataFrame(columns=["Dose", "Selectivity", "Target Bound", "Ligand"])

    for targetPairs in allTargets:
        valencies = [signal[1]]
        targets = []
        naming = []
        for target, valency in targetPairs:
            targets.append(target)
            valencies.append(valency)
            naming.append(f"{target} ({valency})")

        valencies = np.array([valencies])

        dfTargCell = sampleDF.loc[sampleDF["Cell Type"] == targCell]
        targRecs = dfTargCell[[signal[0]] + targets]
        dfOffTargCell = sampleDF.loc[sampleDF["Cell Type"] != targCell]
        offTargRecs = dfOffTargCell[[signal[0]] + targets]

        for dose in doseVec:
            optSelec, optParams = optimize_affs(
                targRecs=targRecs.to_numpy(),
                offTargRecs=offTargRecs.to_numpy(),
                dose=dose,
                valencies=valencies,
            )

            Rbound = get_cell_bindings(
                recCounts=sampleDF[[signal[0]] + targets].to_numpy(),
                monomerAffs=optParams,
                dose=dose,
                valencies=valencies,
            )

            cellBindDF = sampleDF[[signal[0]] + ["Cell Type"]]
            cellBindDF.insert(0, "Receptor Bound", Rbound[:, 0], True)
            cellBindDF = cellBindDF.groupby(["Cell Type"]).mean(0)

            data = {
                "Dose": [dose],
                "Selectivity": 1 / optSelec,
                "Target Bound": cellBindDF["Receptor Bound"].loc[targCell],
                "Ligand": " + ".join(naming),
                "Affinities": optParams,
            }
            df_temp = pd.DataFrame(
                data, columns=["Dose", "Selectivity", "Target Bound", "Ligand"]
            )
            df = df_temp if df.empty else pd.concat([df, df_temp], ignore_index=True)

    sns.lineplot(data=df, x="Dose", y="Selectivity", hue="Ligand", ax=ax[0])
    sns.lineplot(data=df, x="Dose", y="Target Bound", hue="Ligand", ax=ax[1])
    ax[0].set(xscale="log")
    ax[1].set(xscale="log")

    return f
