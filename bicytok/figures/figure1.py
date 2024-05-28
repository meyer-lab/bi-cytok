from os.path import dirname, join
from .common import getSetup
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from ..selectivityFuncs import (
    get_cell_bindings,
    getSampleAbundances,
    optimizeDesign,
)

path_here = dirname(dirname(__file__))

plt.rcParams["svg.fonttype"] = "none"


def makeFigure():
    """Figure file to generate dose response curves for any combination of multivalent and multispecific ligands.
    Outputs dose vs. selectivity for the target cell and amount of target cell bound at indicated signal receptor.
        signal: Receptor that the ligand is delivering signal to; selectivity and target bound are with respect to engagement
    with this receptor
        allTargets: List of paired [(target receptor, valency)] combinations for each targeting receptor; to be used for targeting
    the target cell, not signaling
    """
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
            "CD4 TEM",
            "NK Proliferating",
            "NK_CD56bright",
        ]
    )
    targCell = "Treg"
    offTCells = cells[cells != targCell]

    epitopesList = pd.read_csv(join(path_here, "data/epitopeList.csv"))
    epitopes = list(epitopesList["Epitope"].unique())
    epitopesDF = getSampleAbundances(epitopes, cells)

    doseVec = np.logspace(-2, 2, num=20)
    df = pd.DataFrame(columns=["Dose", "Selectivity", "Target Bound", "Ligand"])

    for targetPairs in allTargets:
        print(targetPairs)
        prevOptAffs = [8.0]
        valencies = [signal[1]]
        targets = []
        naming = []
        for target, valency in targetPairs:
            prevOptAffs.append(8.0)
            targets.append(target)
            valencies.append(valency)
            naming.append("{} ({})".format(target, valency))
        valencies = np.array([valencies])

        for _, dose in enumerate(doseVec):
            optParams = optimizeDesign(
                signal[0],
                targets,
                targCell,
                offTCells,
                epitopesDF,
                dose,
                valencies,
                prevOptAffs,
            )
            prevOptAffs = optParams[1]
            cellBindings = get_cell_bindings(
                epitopesDF, signal[0], targets, prevOptAffs, dose, valencies
            )

            data = {
                "Dose": [dose],
                "Selectivity": 1 / optParams[0],
                "Target Bound": cellBindings["Receptor Bound"].loc[targCell],
                "Ligand": " + ".join(naming),
                "Affinities": optParams[1],
            }
            df_temp = pd.DataFrame(
                data, columns=["Dose", "Selectivity", "Target Bound", "Ligand"]
            )
            df = pd.concat([df, df_temp], ignore_index=True)

    print(df)

    sns.lineplot(data=df, x="Dose", y="Selectivity", hue="Ligand", ax=ax[0])
    sns.lineplot(data=df, x="Dose", y="Target Bound", hue="Ligand", ax=ax[1])
    ax[0].set(xscale="log")
    ax[1].set(xscale="log")

    return f
