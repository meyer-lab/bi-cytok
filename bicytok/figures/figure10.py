import pandas as pd
import seaborn as sns
import numpy as np

from .common import getSetup, Figure
from ..selectivityFuncs import getSampleAbundances, optimizeDesign, minSelecFunc, get_rec_vecs


def makeFigure() -> Figure:
    """Get a list of the axis objects and create a figure"""
    il2 = ['CD122', 'CD25', 8.222, 7.65072247, 9.14874165]
    il7 = ['CD127', None, 9.14874165, 7.14266751, None]

    secondary = il2[0]
    epitope = il2[1]
    valency = 2
    wtIL2RaAff = il2[2]
    wtSecondaryAff = il2[3]
    wtEpitopeAff = il2[4]

    epitopesList = pd.read_csv("./bicytok/data/epitopeList.csv")
    epitopes = list(epitopesList['Epitope'].unique())

    targCell = 'Treg'
    cells = ['CD8 Naive', 'NK', 'CD8 TEM', 'CD4 Naive', 'CD4 CTL', 'CD8 TCM', 'CD8 Proliferating',
    'Treg', 'CD4 TEM', 'NK Proliferating', 'NK_CD56bright']
    offTCells = cells[cells != targCell]

    doseVec = np.logspace(-3, 3, num=20)
    epitopesDF = getSampleAbundances(epitopes, cells)
    df = pd.DataFrame(columns=['Dose', 'Affinity (IL2Ra)', 'Affinity (secondary)', 'Affinity (epitope)', 'Selectivity', 'Ligand'])
    df2 = pd.DataFrame(columns=['Dose', 'Target Bound', 'Ligand'])
    targRecs, offTRecs = get_rec_vecs(epitopesDF, targCell, offTCells, secondary, epitope)

    if secondary == 'CD122':
        prevOptAffs = [8.0, 8.0, 8.0]
    else:
        prevOptAffs = [8.0, 8.0]

    for _, dose in enumerate(doseVec):
        optParams = optimizeDesign(secondary, epitope, targCell, offTCells, epitopesDF, dose, valency, prevOptAffs)
        if secondary == 'CD122':
            LD = minSelecFunc([wtIL2RaAff, wtSecondaryAff, wtEpitopeAff], secondary, epitope, targRecs, offTRecs, dose, valency)
            prevOptAffs = [optParams[1][0], optParams[1][1], optParams[1][2]]
            epitopeAff = optParams[1][2]
        else:
            LD = minSelecFunc([wtIL2RaAff, wtSecondaryAff], secondary, epitope, targRecs, offTRecs, dose, valency)
            prevOptAffs = [optParams[1][0], optParams[1][1]]
            epitopeAff = None

        data = {'Dose': [dose],
            'Affinity (IL2Ra)': optParams[1][0],
            'Affinity (secondary)': optParams[1][1],
            'Affinity (epitope)': epitopeAff,
            'Selectivity': 1 / optParams[0],
            'Ligand': "Optimized"
        }

        df_temp = pd.DataFrame(data, columns=['Dose', 'Affinity (IL2Ra)', 'Affinity (secondary)', 'Affinity (epitope)', 'Selectivity', 'Ligand'])
        df = pd.concat([df, df_temp], ignore_index=True)

        data = {'Dose': [dose],
            'Affinity (IL2Ra)': wtIL2RaAff,
            'Affinity (secondary)': wtSecondaryAff,
            'Affinity (epitope)': wtEpitopeAff,
            'Selectivity': 1 / LD,
            'Ligand': "Live/Dead"
        }

        df_temp = pd.DataFrame(data, columns=['Dose', 'Affinity (IL2Ra)', 'Affinity (secondary)', 'Affinity (epitope)', 'Selectivity', 'Ligand'])
        df = pd.concat([df, df_temp], ignore_index=True)

        data = {'Dose': [dose],
            'Target Bound': optParams[2],
            'Ligand': "Optimized"
        }

        df_temp = pd.DataFrame(data, columns=['Dose', 'Target Bound', 'Ligand'])
        df2 = pd.concat([df2, df_temp], ignore_index=True)

        data = {'Dose': [dose],
            'Target Bound': minSelecFunc.targetBound,
            'Ligand': "Live/Dead"
        }

        df_temp = pd.DataFrame(data, columns=['Dose', 'Target Bound', 'Ligand'])
        df2 = pd.concat([df2, df_temp], ignore_index=True)

    if secondary == 'CD122':
        ax, f = getSetup((12, 3), (1, 5))
        sns.lineplot(data=df, x='Dose', y='Affinity (epitope)', hue='Ligand', ax=ax[4])
        ax[4].set(xscale='log')
    else:
        ax, f = getSetup((12, 3), (1, 4))

    sns.lineplot(data=df, x='Dose', y='Selectivity', hue='Ligand', ax=ax[0])
    sns.lineplot(data=df2, x='Dose', y='Target Bound', hue='Ligand', ax=ax[1])
    sns.lineplot(data=df, x='Dose', y='Affinity (IL2Ra)', hue='Ligand', ax=ax[2])
    sns.lineplot(data=df, x='Dose', y='Affinity (secondary)', hue='Ligand', ax=ax[3])

    ax[0].set(xscale='log')
    ax[1].set(xscale='log')
    ax[2].set(xscale='log')
    ax[3].set(xscale='log')

    return f
