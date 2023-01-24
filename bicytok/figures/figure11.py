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
    ax, f = getSetup((8, 3), (1, 2))

    cells = ['CD8 Naive', 'NK', 'CD8 TEM', 'CD4 Naive', 'CD4 CTL', 'CD8 TCM',
    'Treg', 'CD4 TEM', 'NK Proliferating', 'NK_CD56bright']

    epitopesList = pd.read_csv(join(path_here, "data/epitopeList.csv"))
    epitopes = list(epitopesList['Epitope'].unique())
    epitopesDF = getSampleAbundances(epitopes, cells)

    bindings, totalIL2Rb = get_cell_bindings([8.5, 6.0, 8.5], cells, epitopesDF, epitopes[17], 0.1, False)
    allCellTypes = {'Cell Type': ['All Cell Types'],
        'IL2Rb Bound': [bindings['IL2Rb Bound'].sum()]
    }
    bindings = bindings.append(pd.DataFrame(allCellTypes, columns=['Cell Type', 'IL2Rb Bound']))
    bindings['Percent Bound of Total IL2Rb'] = (bindings['IL2Rb Bound'] / totalIL2Rb) * 100
    print(bindings)

    palette = sns.color_palette("husl", 10)
    sns.barplot(data=bindings[:(len(bindings)-1)], x='Cell Type', y='IL2Rb Bound', palette=palette, ax=ax[0])
    sns.barplot(data=bindings, x='Cell Type', y='Percent Bound of Total IL2Rb', palette=palette, ax=ax[1])
    ax[0].set_xticklabels(labels=ax[0].get_xticklabels(), rotation=45, horizontalalignment='right')
    ax[1].set_xticklabels(labels=ax[1].get_xticklabels(), rotation=45, horizontalalignment='right')

    return f
