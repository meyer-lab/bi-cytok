from os.path import dirname, join

import numpy as np
import pandas as pd
import seaborn as sns

from ..selectivityFuncs import (
    getSampleAbundances,
    optimizeDesign,
)
from .common import getSetup

path_here = dirname(dirname(__file__))


def makeFigure():
    """Figure file to generate bispecific ligand selectivity heatmap of selectivity for each bispecific pairing."""
    ax, f = getSetup((4, 3), (1, 1))

    signal = ["CD122", 1]
    allTargets = [("CD25", 1), ("CD278", 1), ("CD45RB", 1), ("CD4-2", 1), ("CD81", 1)]
    dose = 10e-2

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

    df = pd.DataFrame(columns=["Target 1", "Target 2", "Selectivity"])

    valencies = []
    targets = []
    for target, valency in allTargets:
        targets.append(target)
        valencies.append(valency)

    for i, target1 in enumerate(targets):
        for j, target2 in enumerate(targets):
            if i == j:
                targetsBoth = [target1]
                optAffs = [8.0, 8.0]
                valenciesBoth = np.array([[signal[1], valencies[i]]])
            else:
                targetsBoth = [target1, target2]
                optAffs = [8.0, 8.0, 8.0]
                valenciesBoth = np.array([[signal[1], valencies[i], valencies[j]]])

            optParams = optimizeDesign(
                signal[0],
                targetsBoth,
                targCell,
                offTCells,
                epitopesDF,
                dose,
                valenciesBoth,
                optAffs,
            )

            data = {
                "Target 1": f"{target1} ({valencies[i]})",
                "Target 2": f"{target2} ({valencies[j]})",
                "Selectivity": 1 / optParams[0],
            }
            df_temp = pd.DataFrame(
                data, columns=["Target 1", "Target 2", "Selectivity"], index=[0]
            )
            df = pd.concat([df, df_temp], ignore_index=True)

    selectivities = df.pivot(index="Target 1", columns="Target 2", values="Selectivity")
    sns.heatmap(selectivities)

    return f
