from os.path import dirname, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..selectivityFuncs import (
    calcReceptorAbundances,
    get_cell_bindings,
    optimizeSelectivityAffs,
)
from .common import getSetup

path_here = dirname(dirname(__file__))

plt.rcParams["svg.fonttype"] = "none"


"""SIGNAL: Receptor that the ligand is delivering signal to; selectivity and
    target bound are with respect to engagement with this receptor
ALL_TARGETS: List of paired [(target receptor, valency)] combinations for each
    targeting receptor; to be used for targeting
the target cell, not signaling
CELLS: Array of cells that will be sampled from and used in calculations
TARG_CELL: Target cell whose selectivity will be maximized
STARTING_AFF: Starting affinity to modulate from in order to
    maximize selectivity for the targCell, affinities are K_a in L/mol"""

SIGNAL = ["CD122", 1]
ALL_TARGETS = [
    [("CD25", 1)],
    [("CD25", 4)],
    [("CD25", 1), ("CD278", 1)],
    [("CD25", 4), ("CD278", 4)],
    [("CD25", 1), ("CD27", 1)],
    [("CD25", 4), ("CD27", 4)],
    [("CD25", 1), ("CD278", 1), ("CD27", 1)],
    [("CD25", 4), ("CD278", 4), ("CD27", 4)],
]

CELLS = np.array(
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
TARG_CELL = "Treg"

STARTING_AFF = 8.0


def makeFigure():
    """Figure file to generate dose response curves for any combination of
    multivalent and multispecific ligands.
    Outputs dose vs. selectivity for the target cell and amount of target cell
    bound at indicated signal receptor."""
    ax, f = getSetup((6, 3), (1, 2))

    offTargCells = CELLS[CELLS != TARG_CELL]

    epitopesList = pd.read_csv(join(path_here, "data/epitopeList.csv"))
    epitopes = list(epitopesList["Epitope"].unique())
    epitopesDF = calcReceptorAbundances(epitopes, CELLS)

    doseVec = np.logspace(-2, 2, num=20)
    df = pd.DataFrame(columns=["Dose", "Selectivity", "Target Bound", "Ligand"])

    for targetPairs in ALL_TARGETS:
        optimizedAffs = [STARTING_AFF]
        valencies = [SIGNAL[1]]
        targets = []
        naming = []
        for target, valency in targetPairs:
            optimizedAffs.append(STARTING_AFF)
            targets.append(target)
            valencies.append(valency)
            naming.append("{} ({})".format(target, valency))
        valencies = np.array([valencies])

        for _, dose in enumerate(doseVec):
            optParams = optimizeSelectivityAffs(
                SIGNAL[0],
                targets,
                TARG_CELL,
                offTargCells,
                epitopesDF,
                dose,
                valencies,
                optimizedAffs,
            )
            optimizedAffs = optParams[1]
            cellBindings = get_cell_bindings(
                epitopesDF, SIGNAL[0], targets, optimizedAffs, dose, valencies
            )

            data = {
                "Dose": [dose],
                "Selectivity": 1 / optParams[0],
                "Target Bound": cellBindings["Receptor Bound"].loc[TARG_CELL],
                "Ligand": " + ".join(naming),
                "Affinities": optParams[1],
            }
            df_temp = pd.DataFrame(
                data, columns=["Dose", "Selectivity", "Target Bound", "Ligand"]
            )
            df = pd.concat([df, df_temp], ignore_index=True)

    sns.lineplot(data=df, x="Dose", y="Selectivity", hue="Ligand", ax=ax[0])
    sns.lineplot(data=df, x="Dose", y="Target Bound", hue="Ligand", ax=ax[1])
    ax[0].set(xscale="log")
    ax[1].set(xscale="log")

    return f
