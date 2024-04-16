from os.path import dirname, join
from .common import getSetup
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from ..selectivityFuncs import get_cell_bindings, getSampleAbundances, get_rec_vecs, optimizeDesign, minSelecFunc
from ..imports import importCITE

path_here = dirname(dirname(__file__))

def makeFigure():
    """Figure to generate selectivity bar plots for any combination of multivalent and multispecific ligand."""
    ax, f = getSetup((4, 3), (1, 1))

    signal = ['CD122', 1]
    allTargets = [[('CD25', 1)], [('CD278', 1)], [('CD25', 1), ('CD278', 1)], [('CD278', 1), ('CD45RB', 1)],
        [('CD278', 1), ('CD81', 1)], [('CD278', 1), ('CD4-2', 1)], [('CD25', 1), ('CD278', 1), ('CD45RB', 1)],
        [('CD25', 1), ('CD278', 1), ('CD81', 1)], [('CD25', 1), ('CD278', 1), ('CD4-2', 1)],
        [('CD25', 4)], [('CD278', 4)], [('CD25', 4), ('CD278', 4)], [('CD278', 4), ('CD45RB', 4)],
        [('CD278', 4), ('CD81', 4)], [('CD278', 4), ('CD4-2', 4)], [('CD25', 4), ('CD278', 4), ('CD45RB', 4)],
        [('CD25', 4), ('CD278', 4), ('CD81', 4)], [('CD25', 4), ('CD278', 4), ('CD4-2', 4)]]
    dose = 10e-1

    cells = np.array(['CD8 Naive', 'NK', 'CD8 TEM', 'CD4 Naive', 'CD4 CTL', 'CD8 TCM', 'CD8 Proliferating',
        'Treg', 'CD4 TEM', 'NK Proliferating', 'NK_CD56bright'])
    targCell = 'Treg'
    offTCells = cells[cells != targCell]

    epitopesList = pd.read_csv(join(path_here, "data/epitopeList.csv"))
    epitopes = list(epitopesList['Epitope'].unique())
    epitopesDF = getSampleAbundances(epitopes, cells)

    df = pd.DataFrame(columns=['Selectivity', 'Ligand'])

    for targetPairs in allTargets:
        optAffs = [8.0, 8.0, 8.0]

        valencies = [signal[1]]
        targets = []
        naming = []
        for target, valency in targetPairs:
            targets.append(target)
            valencies.append(valency)
            naming.append('{} ({})'.format(target, valency))

        optParams = optimizeDesign(signal[0], targets, targCell, offTCells, epitopesDF, dose, valencies, optAffs)

        data = {'Selectivity': 1 / optParams[0],
            'Ligand': ' + '.join(naming)
        }
        df_temp = pd.DataFrame(data, columns=['Selectivity', 'Ligand'], index=[0])
        df = pd.concat([df, df_temp], ignore_index=True)

    palette = sns.color_palette("husl", 10)
    sns.barplot(data=df, x='Ligand', y='Selectivity', palette=palette, ax=ax[0])
    ax[0].set_xticklabels(labels=ax[0].get_xticklabels(), rotation=45, horizontalalignment='right')

    return f