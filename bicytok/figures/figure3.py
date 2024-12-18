from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

from ..selectivityFuncs import (
    sampleReceptorAbundances,
    get_cell_bindings,
)
from .common import getSetup
from ..imports import importCITE

path_here = Path(__file__).parent.parent


def makeFigure():
    """
    Figure file to generate bar plots for amount of signal receptor bound to each given cell type
    signal: signaling receptor
    target: additional targeting receptor
    signalAff: starting affinity of ligand and signal receptor
    """

    ax, f = getSetup((8, 3), (1, 2))

    signal = ["CD122"]
    targets = ["CD25", "CD278"]

    # Armaan: how did we pick this affinity?
    signalAff = 6.0
    # Armaan: how are the 8.5s chosen here?
    targetAffs = [8.5, 8.5]
    valency = 4
    dose = 0.1

    affs = np.array([signalAff] + targetAffs)
    valencies = np.array([[valency, valency, valency]])

    cellTypes = [
        "Treg",
        "CD8 Naive",
        "NK",
        "CD8 TEM",
        "CD4 Naive",
        "CD4 CTL",
        "CD8 TCM",
        "CD4 TEM",
        "NK Proliferating",
        "NK_CD56bright",
    ]
    
    epitopesList = pd.read_csv(
        path_here / "data" / "epitopeList.csv"
    )
    epitopes = list(epitopesList["Epitope"].unique())

    CITE_DF = importCITE()
    epitopesDF = CITE_DF[epitopes + ["CellType2"]]
    epitopesDF = epitopesDF.loc[epitopesDF["CellType2"].isin(cellTypes)]
    epitopesDF = epitopesDF.rename(columns={"CellType2": "Cell Type"})

    sampleDF = sampleReceptorAbundances(
        CITE_DF=epitopesDF,
        epitopes=epitopes,
        numCells=1000
    )
    
    Rbound = get_cell_bindings(
        recCounts = sampleDF[signal + targets].to_numpy(),
        monomerAffs = affs,
        dose = dose,
        valencies = valencies,
    )

    cellBindDF = sampleDF[signal + ["Cell Type"]]
    cellBindDF.insert(0, "Receptor Bound", Rbound[:, 0], True)
    cellBindDF = cellBindDF.groupby(["Cell Type"]).mean(0)
    cellBindDF["Percent Bound of Signal Receptor"] = (
        cellBindDF["Receptor Bound"] / cellBindDF[signal]
    ) * 10

    palette = sns.color_palette("husl", 10)
    sns.barplot(
        data=Rbound, x=Rbound.index, y="Receptor Bound", palette=palette, ax=ax[0]
    )
    sns.barplot(
        data=Rbound,
        x=Rbound.index,
        y="Percent Bound of Signal Receptor",
        palette=palette,
        ax=ax[1],
    )
    ax[0].set_xticklabels(
        labels=ax[0].get_xticklabels(), rotation=45, horizontalalignment="right"
    )
    ax[1].set_xticklabels(
        labels=ax[1].get_xticklabels(), rotation=45, horizontalalignment="right"
    )

    return f