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
    ax, f = getSetup((4, 3), (1, 1))

    signal = ['CD127', 1]
    allTargets = [[('CD25', 1)], [('CD25', 1), ('CD278', 1)], [('CD25', 1), ('CD278', 1), ('CD4-2', 1)],
        [('CD25', 4)], [('CD25', 4), ('CD278', 4)], [('CD25', 4), ('CD278', 4), ('CD4-2', 4)]]

    cells = np.array(['CD8 Naive', 'NK', 'CD8 TEM', 'CD4 Naive', 'CD4 CTL', 'CD8 TCM', 'CD8 Proliferating',
        'Treg', 'CD4 TEM', 'NK Proliferating', 'NK_CD56bright'])
    targCell = 'Treg'
    offTCells = cells[cells != targCell]

    epitopesList = pd.read_csv(join(path_here, "data/epitopeList.csv"))
    epitopes = list(epitopesList['Epitope'].unique())
    epitopesDF = getSampleAbundances(epitopes, cells)

    dose = 10e-2
    df = pd.DataFrame(columns=['Dose', 'Selectivity', 'Target Bound', 'Ligand'])

    optAffs = [8.0, 8.0, 8.0]

    valencies = []
    targets = []
    for target, valency in allTargets:
        targets.append(target)
        valencies.append(valency)

    for i, target1 in enumerate(targets):
        for j, target2 in enumerate(targets):
            if i == j:
                targetsBoth = [target1]
                valenciesBoth = [signal[1], valencies[i]]
            else:
                targetsBoth = [target1, target2]
                valenciesBoth = [signal[1], valencies[i], valencies[j]]
                
            optParams = optimizeDesign(signal[0], targetsBoth, targCell, offTCells, epitopesDF, dose, valenciesBoth, optAffs)
                
            data = {'Target 1': '{} ({})'.format(target1, valencies[i]),
                'Target 2': '{} ({})'.format(target2, valencies[j]),
                'Selectivity': 1 / optParams[0]
            }
            df_temp = pd.DataFrame(data, columns=['KL Divergence', 'Wasserstein Distance', 'Selectivity'])
            df = pd.concat([df, df_temp], ignore_index=True)

    sns.lineplot(data=df, x='KL Divergence', y='Selectivity', ax=ax[0])
    sns.lineplot(data=df, x='Wasserstein Distance', y='Selectivity', ax=ax[1])
    ax[0].set(xscale='log')
    ax[1].set(xscale='log')

    return f
