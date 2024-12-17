import numpy as np
import pandas as pd
import seaborn as sns

from ..selectivityFuncs import (
    calcReceptorAbundances,
    get_cell_bindings,
)
from .common import getSetup


def makeFigure():
    """
    Figure file to generate bar plots for amount of signal receptor bound to each given cell type
    secondary: signaling receptor
    epitope: additional targeting receptor
    Armaan: why call it 'starting' affinity? I don't think you're fitting any
    affinities in this figure, so maybe state this explicitly and just call them
    affinities. Also, are you sure you shouldn't be optimizing the affinities here?
    This seems to be the case for the other figures.
    secondaryAff: starting affinity of ligand and secondary receptor
    """
    ax, f = getSetup((8, 3), (1, 2))

    # Armaan: Why call it secondary? Can we use a better name?
    secondary = "CD122"
    epitope = "CD278"
    # Armaan: how did we pick this affinity?
    secondaryAff = 6.0
    valency = 4

    # Armaan: how are the 8.5s chosen here? I think it would be best to declare
    # a lot of these literals at the top of the file.
    affs = np.array([secondaryAff, 8.5, 8.5])

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
    
    # Armaan: use os.path.join or pathlib.Path here
    epitopesList = pd.read_csv("./bicytok/data/epitopeList.csv")
    epitopes = list(epitopesList["Epitope"].unique())

    epitopesDF = calcReceptorAbundances(epitopes, cells)

    # Armaan: Why is CD25 declared down here while EPITOPE is declared at
    # the top of the file?
    bindings = get_cell_bindings(
        epitopesDF,
        secondary,
        ["CD25", epitope],
        affs,
        0.1,
        np.array([[valency, valency, valency]]),
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