from os.path import dirname, join
from .common import getSetup, KL_divergence_2D_pair, EMD_2D_pair, correlation
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from ..selectivityFuncs import get_cell_bindings, getSampleAbundances, get_rec_vecs, optimizeDesign, minSelecFunc
from ..imports import importCITE

path_here = dirname(dirname(__file__))

def makeFigure():
    ax, f = getSetup((9, 3), (1, 3))

    markerDF = importCITE()
    new_df = markerDF.head(1000)
# Make it show the curve for different valencies and we'll see how it matters less
    signal = ['CD122', 1]
    allTargets = [[('CD25', 1), ('CD278', 1)], [('CD25', 1), ('CD4-2', 1)], [('CD25', 1), ('CD45RB', 1)], [('CD25', 1), ('CD81', 1)],
        [('CD278', 1), ('CD4-2', 1)], [('CD278', 1), ('CD45RB', 1)], [('CD278', 1), ('CD81', 1)], [('CD4-2', 1), ('CD45RB', 1)],
        [('CD4-2', 1), ('CD81', 1)], [('CD45RB', 1), ('CD81', 1)]]
    dose = 10e-2

    cells = np.array(['CD8 Naive', 'NK', 'CD8 TEM', 'CD4 Naive', 'CD4 CTL', 'CD8 TCM', 'CD8 Proliferating',
        'Treg', 'CD4 TEM', 'NK Proliferating', 'NK_CD56bright'])
    targCell = 'Treg'
    offTCells = cells[cells != targCell]

    epitopesList = pd.read_csv(join(path_here, "data/epitopeList.csv"))
    epitopes = list(epitopesList['Epitope'].unique())
    epitopesDF = getSampleAbundances(epitopes, cells)

    df = pd.DataFrame(columns=['KL Divergence', "Earth Mover's Distance", 'Correlation', 'Selectivity'])

    for targetPairs in allTargets:
        prevOptAffs = [8.0, 8.0, 8.0]

        valencies = [signal[1]]
        targets = []
        for target, valency in targetPairs:
            targets.append(target)
            valencies.append(valency)

        optParams = optimizeDesign(signal[0], targets, targCell, offTCells, epitopesDF, dose, valencies, prevOptAffs)
        prevOptAffs = [optParams[1][0], optParams[1][1], optParams[1][2]]

        KLD = KL_divergence_2D_pair(new_df, targCell, targets[0], targets[1])
        EMD = EMD_2D_pair(new_df, targCell, targets[0], targets[1])
        corr = correlation(targCell, targets).loc[targets[0], targets[1]]
            
        data = {'KL Divergence': [KLD],
            "Earth Mover's Distance": [EMD],
            'Correlation': [corr],
            'Selectivity': 1 / optParams[0]
        }
        df_temp = pd.DataFrame(data, columns=['KL Divergence', "Earth Mover's Distance", 'Correlation', 'Selectivity'])
        df = pd.concat([df, df_temp], ignore_index=True)

    sns.lineplot(data=df, x='KL Divergence', y='Selectivity', ax=ax[0])
    sns.lineplot(data=df, x="Earth Mover's Distance", y='Selectivity', ax=ax[1])
    sns.lineplot(data=df, x='Correlation', y='Selectivity', ax=ax[2])
    ax[0].set(xscale='log')
    ax[1].set(xscale='log')

    return f
