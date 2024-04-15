from .common import getSetup
import pandas as pd
import seaborn as sns
import numpy as np
from ..selectivityFuncs import get_cell_bindings, getSampleAbundances, get_rec_vecs, optimizeDesign, minSelecFunc


def makeFigure():
    ax, f = getSetup((12, 6), (2, 10), multz={0:2, 5:2, 7:1, 10:1, 12:1, 14:1, 16:1, 18:1})

    secondary = 'CD122'
    epitope = 'CD278'
    secondaryAff = 6.0
    valency = 2

    wtIL2RaAff = 8.222
    wtSecondaryAff = 7.65072247
    wtEpitopeAff = 9.14874165

    cells1 = ['Treg', 'CD8 Naive', 'NK', 'CD8 TEM', 'CD4 Naive', 'CD4 CTL', 'CD8 TCM', 'CD4 TEM', 'NK Proliferating',
    'NK_CD56bright']
    cells2 = ['Treg Memory', 'Treg Naive']

    epitopesList = pd.read_csv("./bicytok/data/epitopeList.csv")
    epitopes = list(epitopesList['Epitope'].unique())

    epitopesDF1 = getSampleAbundances(epitopes, cells1, "CellType2")
    epitopesDF2 = getSampleAbundances(epitopes, cells2, "CellType3")

    affs = np.array([[8.5, secondaryAff, 8.5]])
    bindings1 = get_cell_bindings(affs, cells1, epitopesDF1, secondary, epitope, 0.1, valency)
    bindings1['Percent Bound of Secondary'] = (bindings1['Secondary Bound'] / bindings1['Total Secondary']) * 100

    bindings2 = get_cell_bindings(affs, cells2, epitopesDF2, secondary, epitope, 0.1, valency)
    bindings2['Percent Bound of Secondary'] = (bindings2['Secondary Bound'] / bindings2['Total Secondary']) * 100

    palette = sns.color_palette("husl", 10)
    sns.barplot(data=bindings1, x='Cell Type', y='Secondary Bound', palette=palette, ax=ax[0])
    sns.barplot(data=bindings2, x='Cell Type', y='Secondary Bound', palette=palette, ax=ax[1])
    sns.barplot(data=bindings1, x='Cell Type', y='Percent Bound of Secondary', palette=palette, ax=ax[3])
    sns.barplot(data=bindings2, x='Cell Type', y='Percent Bound of Secondary', palette=palette, ax=ax[4])
    ax[0].set_xticklabels(labels=ax[0].get_xticklabels(), rotation=45, horizontalalignment='right')
    ax[1].set_xticklabels(labels=ax[1].get_xticklabels(), rotation=45, horizontalalignment='right')
    ax[3].set_xticklabels(labels=ax[3].get_xticklabels(), rotation=45, horizontalalignment='right')
    ax[4].set_xticklabels(labels=ax[4].get_xticklabels(), rotation=45, horizontalalignment='right')


    targCell = 'Treg Memory'
    cells = ['Treg Memory', 'Treg Naive', 'CD8 Naive', 'NK_2', 'CD8 TEM_1', 'CD4 Naive', 'CD4 CTL', 'CD4 TCM_3', 'CD4 TCM_2',
    'CD8 TCM_1', 'NK_4', 'Treg Naive', 'CD4 TEM_1', 'CD4 TEM_3', 'NK Proliferating', 'CD8 TEM_4', 'NK_CD56bright', 'CD4 TEM_4', 'NK_3']
    offTCells = [c for c in cells if c != targCell]

    doseVec = np.logspace(-3, 3, num=20)
    epitopesDF = getSampleAbundances(epitopes, cells, "CellType3")
    df = pd.DataFrame(columns=['Dose', 'Affinity (IL2Ra)', 'Affinity (secondary)', 'Affinity (epitope)', 'Selectivity', 'Ligand'])
    df2 = pd.DataFrame(columns=['Dose', 'Target Bound', 'Ligand'])
    targRecs, offTRecs = get_rec_vecs(epitopesDF, targCell, offTCells, secondary, epitope)

    for _, dose in enumerate(doseVec):
        optParams = optimizeDesign(secondary, epitope, targCell, offTCells, epitopesDF, dose, valency)
        
        affs = np.array([[wtIL2RaAff, wtSecondaryAff, wtEpitopeAff]])
        LD = minSelecFunc(affs, secondary, epitope, targRecs, offTRecs, dose, valency)
        epitopeAff = optParams[1][2]

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

    sns.lineplot(data=df, x='Dose', y='Selectivity', hue='Ligand', ax=ax[6])
    sns.lineplot(data=df2, x='Dose', y='Target Bound', hue='Ligand', ax=ax[7])
    sns.lineplot(data=df, x='Dose', y='Affinity (IL2Ra)', hue='Ligand', ax=ax[8])
    sns.lineplot(data=df, x='Dose', y='Affinity (secondary)', hue='Ligand', ax=ax[9])
    sns.lineplot(data=df, x='Dose', y='Affinity (epitope)', hue='Ligand', ax=ax[10])

    ax[6].set(xscale='log')
    ax[7].set(xscale='log')
    ax[8].set(xscale='log')
    ax[9].set(xscale='log')
    ax[10].set(xscale='log')

    ax[2].axis("off")
    ax[5].axis("off")

    return f
