from os.path import dirname, join
from .common import getSetup
import pandas as pd
import seaborn as sns
import numpy as np
import scipy

from ..selectivityFuncs import getSampleAbundances, optimizeDesign, minSelecFunc
from ..imports import importCITE

path_here = dirname(dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    ax, f = getSetup((4, 5), (3, 1))

    epitopesList = pd.read_csv(join(path_here, "data/epitopeList.csv"))
    epitopes = list(epitopesList['Epitope'].unique())

    targCell = 'Treg'
    cells = np.array(['CD14 Mono', 'CD4 TCM', 'CD8 Naive', 'NK', 'CD8 TEM', 'CD16 Mono',
    'B intermediate', 'CD4 Naive', 'CD4 CTL', 'B naive', 'CD8 TCM', 'B memory', 'CD8 Proliferating', 
    'Treg', 'CD4 TEM', 'NK Proliferating', 
    'NK_CD56bright', 'CD4 Proliferating', 'ILC'])
    offTCells = cells[cells != targCell]

    doseVec = np.logspace(-3, 5, num=1) #num=20
    epitopesDF = getSampleAbundances(epitopes, cells)
    df = pd.DataFrame(columns=['Epitope', 'Dose', 'Affinity 1', 'Affinity 2', 'Affinity 3'])

    output = np.zeros(doseVec.size)

    for j, dose in enumerate(doseVec):
        optParams = optimizeDesign(targCell, offTCells, epitopesDF, epitopes[17], dose)
        print(optParams)

        data = {'Epitope': epitopes[17],
            'Dose': [dose],
            'Affinity 1': optParams[1][0][0],
            'Affinity 2': optParams[1][1][0],
            'Affinity 3': optParams[1][2][0]
        }

        df2 = pd.DataFrame(data, columns=['Epitope', 'Dose', 'Affinity 1', 'Affinity 2', 'Affinity 3'])
        df = df.append(df2, ignore_index=True)

    sns.lineplot(data=df, x='Dose', y='Affinity 1', ax=ax[0])
    sns.lineplot(data=df, x='Dose', y='Affinity 2', ax=ax[1])
    sns.lineplot(data=df, x='Dose', y='Affinity 3', ax=ax[2])
    ax[0].set(xscale='log')
    ax[1].set(xscale='log')
    ax[2].set(xscale='log')

    return f
