from os.path import dirname, join
from .common import getSetup
import pandas as pd
import seaborn as sns
import numpy as np
import scipy

from ..selectivityFuncs import getSampleAbundances, optimizeDesign, minSelecFunc, get_rec_vecs
from ..imports import importCITE

path_here = dirname(dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    ax, f = getSetup((6, 5), (2, 2))

    epitopesList = pd.read_csv(join(path_here, "data/epitopeList.csv"))
    epitopes = list(epitopesList['Epitope'].unique())

    targCell = 'Treg'
    cells = np.array(['CD14 Mono', 'CD4 TCM', 'CD8 Naive', 'NK', 'CD8 TEM', 'CD16 Mono',
    'B intermediate', 'CD4 Naive', 'CD4 CTL', 'B naive', 'CD8 TCM', 'B memory', 'CD8 Proliferating', 
    'Treg', 'CD4 TEM', 'NK Proliferating', 'NK_CD56bright', 'CD4 Proliferating', 'ILC'])
    offTCells = cells[cells != targCell]

    doseVec = np.logspace(-3, 3, num=20)
    epitopesDF = getSampleAbundances(epitopes, cells)
    df = pd.DataFrame(columns=['Epitope', 'Dose', 'Affinity (IL2Ra)', 'Affinity (IL2Rb)', 'Affinity (epitope)', 'Selectivity'])
    targRecs, offTRecs = get_rec_vecs(epitopesDF, targCell, offTCells, epitopes[17])

    for _, dose in enumerate(doseVec):
        optParams = optimizeDesign(targCell, offTCells, epitopesDF, epitopes[17], dose)
        LD = minSelecFunc([8.222, 7.65072247, 9.14874165], targRecs, offTRecs, dose, True)

        data = {'Epitope': epitopes[17],
            'Dose': [dose],
            'Affinity (IL2Ra)': optParams[1][0],
            'Affinity (IL2Rb)': optParams[1][1],
            'Affinity (epitope)': optParams[1][2],
            'Selectivity': optParams[0],
            'Ligand': "Optimized"
        }

        df2 = pd.DataFrame(data, columns=['Epitope', 'Dose', 'Affinity (IL2Ra)', 'Affinity (IL2Rb)', 'Affinity (epitope)', 'Selectivity'])
        df = df.append(df2, ignore_index=True)

        data = {'Epitope': epitopes[17],
            'Dose': [dose],
            'Affinity (IL2Ra)': 8.222,
            'Affinity (IL2Rb)': 7.65072247,
            'Affinity (epitope)': 9.14874165,
            'Selectivity': LD,
            'Ligand': "Live/Dead"
        }

        df3 = pd.DataFrame(data, columns=['Epitope', 'Dose', 'Affinity (IL2Ra)', 'Affinity (IL2Rb)', 'Affinity (epitope)', 'Selectivity', 'Ligand'])
        df = df.append(df3, ignore_index=True)

    sns.lineplot(data=df, x='Dose', y='Affinity (IL2Ra)', hue='Ligand', ax=ax[0])
    sns.lineplot(data=df, x='Dose', y='Affinity (IL2Rb)', hue='Ligand', ax=ax[1])
    sns.lineplot(data=df, x='Dose', y='Affinity (epitope)', hue='Ligand', ax=ax[2])
    sns.lineplot(data=df, x='Dose', y='Selectivity', hue='Ligand', ax=ax[3])
    ax[0].set(xscale='log')
    ax[1].set(xscale='log')
    ax[2].set(xscale='log')
    ax[3].set(xscale='log')

    return f
