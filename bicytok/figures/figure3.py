from os.path import dirname, join

import numpy as np
import pandas as pd
import seaborn as sns
import os

from ..selectivityFuncs import (
    calcReceptorAbundances,
    get_cell_bindings,
)
from .common import getSetup

path_here = dirname(dirname(__file__))



def makeFigure():
    """
    Figure file to generate bar plots for amount of signal receptor bound to each given cell type
    signal: signaling receptor
    target: additional targeting receptor
    Armaan: why call it 'starting' affinity? I don't think you're fitting any
    affinities in this figure, so maybe state this explicitly and just call them
    affinities. Also, are you sure you shouldn't be optimizing the affinities here?
    This seems to be the case for the other figures.
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

    affs = np.array([signalAff] + targetAffs)
    valencies = np.array([[valency, valency, valency]])

    cells = [
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
    
    epitopesList = pd.read_csv(join(path_here, "data", "epitopeList.csv"))
    epitopes = list(epitopesList["Epitope"].unique())

    epitopesDF = calcReceptorAbundances(epitopes, cells)
    
    Rbound = get_cell_bindings(
        epitopesDF[signal + targets].to_numpy(),
        affs,
        0.1,
        valencies,
    )
    Rbound["Percent Bound of Signal Receptor"] = (
        Rbound["Receptor Bound"] / Rbound[signal]
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