from os.path import dirname, join
from .common import getSetup, correlation
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from ..distanceMetricFuncs import KL_divergence_2D_pair, EMD_2D_pair


from scipy.optimize import least_squares
from ..selectivityFuncs import get_cell_bindings, getSampleAbundances, get_rec_vecs, optimizeDesign, minSelecFunc
from ..imports import importCITE
from random import sample, seed

path_here = dirname(dirname(__file__))

# NOTE: Make this work still
def makeFigure():
    """KL divergence, EMD, and anti-correlation correlation with selectivity at a given dose."""
    ax, f = getSetup((9, 3), (1, 3))

    CITE_DF = importCITE()
    new_df = CITE_DF.sample(10000, random_state=42)

    signal_receptor = 'CD122'
    signal_valency = 1
    valencies = [1, 2, 4]
    allTargets = [['CD25', 'CD278'], ['CD25', 'CD4-2'], ['CD25', 'CD45RB'], ['CD25', 'CD81'], ['CD278', 'CD4-2'],['CD278', 'CD45RB'], ['CD278', 'CD81'], ['CD4-2', 'CD45RB'], ['CD4-2', 'CD81'], ['CD45RB', 'CD81']]
    dose = 10e-2
    cells = np.array(['CD8 Naive', 'NK', 'CD8 TEM', 'CD4 Naive', 'CD4 CTL', 'CD8 TCM', 'CD8 Proliferating','Treg', 'CD4 TEM', 'NK Proliferating', 'NK_CD56bright'])
    targCell = 'Treg'
    offTCells = cells[cells != targCell]

    epitopesList = pd.read_csv(join(path_here, "data/epitopeList.csv"))
    epitopes = list(epitopesList['Epitope'].unique())
    epitopesDF = getSampleAbundances(epitopes, cells, numCells=10000)

    targetSize = 30 
    i = len(allTargets)
    while i < targetSize:
        targs = sample(epitopes, 2)
        if not targs in allTargets:
            allTargets.append(targs)
            i += 1

    df = pd.DataFrame(columns=['KL Divergence', "Earth Mover's Distance", 'Correlation', 'Selectivity', 'Valency'])

    for val in valencies:
        for targets in allTargets:
            prevOptAffs = [8.0, 8.0, 8.0]

            vals = [signal_valency, val, val]

            optParams = optimizeDesign(signal_receptor, targets, targCell, offTCells, epitopesDF, dose, vals, prevOptAffs)
            prevOptAffs = [optParams[1][0], optParams[1][1], optParams[1][2]]

            KLD = KL_divergence_2D_pair(new_df, targCell, targets[0], targets[1])
            EMD = EMD_2D_pair(new_df, targCell, targets[0], targets[1])
            corr = correlation(targCell, targets).loc[targets[0], targets[1]]

            data = {'KL Divergence': [KLD],
                "Earth Mover's Distance": [EMD],
                'Correlation': [corr],
                'Selectivity': 1 / optParams[0],
                'Valency': val
            }
            df_temp = pd.DataFrame(data, columns=['KL Divergence', "Earth Mover's Distance", 'Correlation', 'Selectivity', 'Valency'])
            df = pd.concat([df, df_temp], ignore_index=True)

    sns.lineplot(data=df, x='KL Divergence', y='Selectivity', hue='Valency', ax=ax[0])
    sns.lineplot(data=df, x="Earth Mover's Distance", y='Selectivity', hue='Valency', ax=ax[1])
    sns.lineplot(data=df, x='Correlation', y='Selectivity', hue='Valency', ax=ax[2])
    ax[0].set(xscale='log')
    ax[1].set(xscale='log')

    return f