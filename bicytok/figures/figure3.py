import numpy as np
import pandas as pd
import seaborn as sns

from ..selectivityFuncs import (
    get_cell_bindings,
    calcReceptorAbundances,
)
from .common import getSetup


"""SECONDARY: signaling receptor
EPITOPE: additional targeting receptor
SECONDARY_AFF: starting affinity of ligand and secondary receptor"""

SECONDARY = "CD122"
EPITOPE = "CD278"
SECONDARY_AFF = 6.0
VALENCY = 4

CELLS = [
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


def makeFigure():
    """Figure file to generate bar plots for amount of signal receptor
    bound to each given cell type"""
    ax, f = getSetup((8, 3), (1, 2))

    affs = np.array([SECONDARY_AFF, 8.5, 8.5])

    epitopesList = pd.read_csv("./bicytok/data/epitopeList.csv")
    epitopes = list(epitopesList["Epitope"].unique())

    epitopesDF = calcReceptorAbundances(epitopes, CELLS)

    bindings = get_cell_bindings(
        epitopesDF,
        SECONDARY,
        ["CD25", EPITOPE],
        affs,
        0.1,
        np.array([[VALENCY, VALENCY, VALENCY]]),
    )
    bindings["Percent Bound of Signal Receptor"] = (
        bindings["Receptor Bound"] / bindings[SECONDARY]
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
