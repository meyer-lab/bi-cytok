from os.path import dirname, join
from .common import getSetup
import pandas as pd
import seaborn as sns
import numpy as np
from ..selectivityFuncs import getSampleAbundances, optimizeDesign, minSelecFunc, get_rec_vecs
from ..imports import importCITE

path_here = dirname(dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    ax, f = getSetup((4, 5), (1, 1))

    epitopesList = pd.read_csv(join(path_here, "data/epitopeList.csv"))
    epitopes = list(epitopesList['Epitope'].unique())

    targCell = 'Treg'
    cells = np.array(['CD14 Mono', 'CD4 TCM', 'CD8 Naive', 'NK', 'CD8 TEM', 'CD16 Mono',
    'B intermediate', 'CD4 Naive', 'CD4 CTL', 'B naive', 'CD8 TCM', 'B memory', 'CD8 Proliferating', 
    'Treg', 'CD4 TEM', 'NK Proliferating', 
    'NK_CD56bright', 'CD4 Proliferating', 'ILC'])
    offTCells = cells[cells != targCell]

    doseVec = np.logspace(-3, 3, num=10)
    epitopesDF = getSampleAbundances(epitopes, cells)
    epitopes = epitopes[0:2]
    df = pd.DataFrame(columns=['Epitope', 'Dose', 'Selectivity'])

    output = np.zeros(doseVec.size)

    for i in range(len(epitopes)):
        
        _, optParams, _ = optimizeDesign("CD122", epitopes[i], targCell, offTCells, epitopesDF, 0.1, 1)
        targRecs, offTRecs = get_rec_vecs(epitopesDF, targCell, offTCells, "CD122", epitopes[i])

        for j, dose in enumerate(doseVec):
            output[j] = 1 / minSelecFunc(optParams, "CD122", epitopes[i], targRecs, offTRecs, dose, 2)

        data = {'Epitope': epitopes[i],
            'Dose': doseVec,
            'Selectivity': output
        }

        df2 = pd.DataFrame(data, columns=['Epitope', 'Dose', 'Selectivity'])
        df = pd.concat([df, df2], ignore_index=True)

    sns.lineplot(data=df, x='Dose', y='Selectivity', hue='Epitope', ax=ax[0])
    ax[0].set(xscale='log')

    return f