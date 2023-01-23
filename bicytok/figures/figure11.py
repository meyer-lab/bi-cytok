"""
This creates Figure 11, used to plot amount of IL2Rb bound to each cell type.
"""
from os.path import dirname, join
from .common import getSetup
import pandas as pd
import seaborn as sns
import numpy as np

from ..selectivityFuncs import get_cell_bindings, getSampleAbundances


path_here = dirname(dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    ax, f = getSetup((4, 3), (1, 1))

    cells = ['CD8 Naive', 'NK', 'CD8 TEM', 'CD4 Naive', 'CD4 CTL', 'CD8 TCM',
    'Treg', 'CD4 TEM', 'NK Proliferating', 'NK_CD56bright']

    epitopesList = pd.read_csv(join(path_here, "data/epitopeList.csv"))
    epitopes = list(epitopesList['Epitope'].unique())
    epitopesDF = getSampleAbundances(epitopes, cells)

    bindings = get_cell_bindings([8.5, 6.0, 8.5], cells, epitopesDF, epitopes[17], 0.1, False)
    print(bindings)

    palette = sns.color_palette("husl", 10)
    sns.barplot(data=bindings, x='Cell Type', y='IL2Rb Bound', palette=palette)
    ax[0].set_xticklabels(labels=ax[0].get_xticklabels(), rotation=45)

    return f
