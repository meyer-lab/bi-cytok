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
    ax, f = getSetup((12, 3), (1, 4))

    secondary = 'CD127'
    secondaryStartAff = 8.0
    secondaryLB = 6.0
    secondaryUB = 9.0
    valency = 2
    wtSecondaryAff = 7.14266751

    epitopesList = pd.read_csv(join(path_here, "data/epitopeList.csv"))
    epitopes = list(epitopesList['Epitope'].unique())

    targCell = 'Treg'
    cells = np.array(['CD8 Naive', 'NK', 'CD8 TEM', 'CD4 Naive', 'CD4 CTL', 'CD8 TCM', 'CD8 Proliferating',
    'Treg', 'CD4 TEM', 'NK Proliferating', 'NK_CD56bright'])
    offTCells = cells[cells != targCell]

    doseVec = np.logspace(-3, 3, num=20)
    epitopesDF = getSampleAbundances(epitopes, cells)
    df = pd.DataFrame(columns=['Dose', 'Affinity (IL2Ra)', 'Affinity (secondary)', 'Selectivity', 'Ligand'])
    df2 = pd.DataFrame(columns=['Dose', 'Target Bound', 'Ligand'])
    targRecs, offTRecs = get_rec_vecs(epitopesDF, targCell, offTCells, secondary)

    prevOptAffs = [8.0, secondaryStartAff]

    for _, dose in enumerate(doseVec):
        optParams = optimizeDesign(targCell, offTCells, epitopesDF, secondary, secondaryLB, secondaryUB, dose, valency, prevOptAffs)
        LD = minSelecFunc([9.14874165, wtSecondaryAff], targRecs, offTRecs, dose, valency, False)

        prevOptAffs = [optParams[1][0], optParams[1][1]]

        data = {'Dose': [dose],
            'Affinity (IL2Ra)': optParams[1][0],
            'Affinity (secondary)': optParams[1][1],
            'Selectivity': 1 / optParams[0],
            'Ligand': "Optimized"
        }

        df_temp = pd.DataFrame(data, columns=['Dose', 'Affinity (IL2Ra)', 'Affinity (secondary)', 'Selectivity', 'Ligand'])
        df = df.append(df_temp, ignore_index=True)

        data = {'Dose': [dose],
            'Affinity (IL2Ra)': 9.14874165,
            'Affinity (secondary)': wtSecondaryAff,
            'Selectivity': 1 / LD,
            'Ligand': "Live/Dead"
        }

        df_temp = pd.DataFrame(data, columns=['Dose', 'Affinity (IL2Ra)', 'Affinity (secondary)', 'Selectivity', 'Ligand'])
        df = df.append(df_temp, ignore_index=True)

        data = {'Dose': [dose],
            'Target Bound': optParams[2],
            'Ligand': "Optimized"
        }

        df_temp = pd.DataFrame(data, columns=['Dose', 'Target Bound', 'Ligand'])
        df2 = df2.append(df_temp, ignore_index=True)

        data = {'Dose': [dose],
            'Target Bound': minSelecFunc.targetBound,
            'Ligand': "Live/Dead"
        }

        df_temp = pd.DataFrame(data, columns=['Dose', 'Target Bound', 'Ligand'])
        df2 = df2.append(df_temp, ignore_index=True)

    sns.lineplot(data=df, x='Dose', y='Affinity (IL2Ra)', hue='Ligand', ax=ax[0])
    sns.lineplot(data=df, x='Dose', y='Affinity (secondary)', hue='Ligand', ax=ax[1])
    sns.lineplot(data=df, x='Dose', y='Selectivity', hue='Ligand', ax=ax[2])
    sns.lineplot(data=df2, x='Dose', y='Target Bound', hue='Ligand', ax=ax[3])
    ax[0].set(xscale='log')
    ax[1].set(xscale='log')
    ax[2].set(xscale='log')
    ax[3].set(xscale='log')

    return f
