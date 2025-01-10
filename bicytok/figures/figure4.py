from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

from ..imports import importCITE
from ..selectivityFuncs import optimizeSelectivityAffs, sampleReceptorAbundances
from .common import getSetup

path_here = Path(__file__).parent.parent


def makeFigure():
    """
    Figure file to generate bispecific ligand selectivity heatmap 
        of selectivity for each bispecific pairing.
    """
    
    ax, f = getSetup((4, 3), (1, 1))

    signal = ["CD122", 1]
    allTargets = [("CD25", 1), ("CD278", 1), ("CD45RB", 1), ("CD4-2", 1), ("CD81", 1)]
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
            "CD4 TEM",
            "NK Proliferating",
            "NK_CD56bright",
        ]
    )
    targCell = "Treg"
    offTargCells = cellTypes[cellTypes != targCell]

    epitopesList = pd.read_csv(
        path_here / "data" / "epitopeList.csv"
    )
    epitopes = list(epitopesList["Epitope"].unique())

    CITE_DF = importCITE()
    epitopesDF = CITE_DF[epitopes + ["CellType2"]]
    epitopesDF = epitopesDF.loc[epitopesDF["CellType2"].isin(cellTypes)]
    epitopesDF = epitopesDF.rename(columns={"CellType2": "Cell Type"})


    sampleDF = sampleReceptorAbundances(
        CITE_DF = epitopesDF,
        numCells = 1000
    )

    df = pd.DataFrame(columns=["Target 1", "Target 2", "Selectivity"])

    valencies = []
    targets = []
    for target, valency in allTargets:
        targets.append(target)
        valencies.append(valency)

    for i, target1 in enumerate(targets):
        for j, target2 in enumerate(targets):
            if i == j:
                # Armaan: shouldn't the molecule in this case include 1 subunit
                # targeting SIGNAL and 2 subunits corresponding to target1?
                # Right now it's just 1 subunit for target1.
                targetsBoth = [target1]
                valenciesBoth = np.array([[signal[1], valencies[i]]])
            else:
                targetsBoth = [target1, target2]
                valenciesBoth = np.array([[signal[1], valencies[i], valencies[j]]])

            dfTargCell = sampleDF.loc[
                sampleDF["Cell Type"] == targCell
            ]
            targRecs = dfTargCell[[signal[0]] + targetsBoth]
            dfOffTargCell = sampleDF.loc[
                sampleDF["Cell Type"].isin(offTargCells)
            ]
            offTargRecs = dfOffTargCell[[signal[0]] + targetsBoth]

            optSelec, optParams = optimizeSelectivityAffs(
                targRecs = targRecs.to_numpy(),
                offTargRecs = offTargRecs.to_numpy(),
                dose = dose,
                valencies = valenciesBoth
            )

            data = {
                "Target 1": f"{target1} ({valencies[i]})",
                "Target 2": f"{target2} ({valencies[j]})",
                "Selectivity": 1 / optSelec,
            }
            df_temp = pd.DataFrame(
                data, columns=["Target 1", "Target 2", "Selectivity"], index=[0]
            )
            df = pd.concat([df, df_temp], ignore_index=True)

    selectivities = df.pivot(index="Target 1", columns="Target 2", values="Selectivity")
    sns.heatmap(selectivities)

    return f