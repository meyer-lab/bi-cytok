from .common import getSetup
import pandas as pd
import seaborn as sns
import numpy as np
from ..selectivityFuncs import (
    get_cell_bindings,
    getSampleAbundances,
)


def makeFigure():
    """Figure file to generate bar plots for amount of signal receptor bound to each given cell type
        secondary: signaling receptor
        epitope: additional targeting receptor"""
    ax, f = getSetup((8, 3), (1, 2))

    secondary = "CD122"
    epitope = "CD278"
    secondaryAff = 6.0
    valency = 4

    affs = np.array([secondaryAff, 8.5, 8.5])

    targCell = "Treg"
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
    offTCells = [c for c in cells if c != targCell]

    epitopesList = pd.read_csv("./bicytok/data/epitopeList.csv")
    epitopes = list(epitopesList["Epitope"].unique())

    epitopesDF = getSampleAbundances(epitopes, cells)

    bindings = get_cell_bindings(
        epitopesDF, secondary, ["CD25", epitope], affs, 0.1, [valency, valency, valency]
    )
    bindings["Percent Bound of Signal Receptor"] = (
        bindings["Receptor Bound"] / bindings[secondary]
    ) * 10

    palette = sns.color_palette("husl", 10)
    sns.barplot(
        data=bindings, x=bindings.index, y="Receptor Bound", palette=palette, ax=ax[0]
    )
    sns.barplot(
        data=bindings,
        x=bindings.index,
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
