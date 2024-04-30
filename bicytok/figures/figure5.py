from os.path import dirname, join
from .common import getSetup
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from ..distanceMetricFuncs import EMD_2D, correlation, KL_divergence_2D


from scipy.optimize import least_squares
from ..selectivityFuncs import get_cell_bindings, getSampleAbundances, get_rec_vecs, optimizeDesign, minSelecFunc
from ..imports import importCITE
from random import sample, seed

path_here = dirname(dirname(__file__))

def makeFigure():
    """KL divergence, EMD, and anti-correlation with selectivity at a given dose."""
    ax, f = getSetup((9, 3), (1, 3))

    CITE_DF = importCITE()
    new_df = CITE_DF.sample(1000, random_state=42)

    signal_receptor = 'CD122'
    signal_valency = 1
    valencies = [1, 2, 4]
    allTargets = [['CD25', 'CD278'], ['CD25', 'CD4-2'], ['CD25', 'CD45RB']]
    dose = 10e-2
    cells = np.array(['CD8 Naive', 'NK', 'CD8 TEM', 'CD4 Naive', 'CD4 CTL', 'CD8 TCM', 'CD8 Proliferating','Treg'])
    targCell = 'Treg'
    offTCells = cells[cells != targCell]

    epitopesList = pd.read_csv(join(path_here, "data/epitopeList.csv"))
    epitopes = list(epitopesList['Epitope'].unique())
    epitopesDF = getSampleAbundances(epitopes, cells, numCells=1000)

    df = pd.DataFrame(columns=['KL Divergence', "Earth Mover's Distance", 'Correlation', 'Selectivity', 'Valency'])

    for val in valencies:
        prevOptAffs = [8.0, 8.0, 8.0]
        for targets in allTargets:
            vals = [signal_valency, val, val]

            optParams = optimizeDesign(signal_receptor, targets, targCell, offTCells, epitopesDF, dose, vals, prevOptAffs)
            prevOptAffs = optParams[1]
            select = 1 / optParams[0],
            KLD = KL_divergence_2D(new_df, targets[0], targCell, targets[1], ax = None) 
            EMD = EMD_2D(new_df, targets[0], targCell, targets[1], ax = None)
            corr = correlation(targCell, targets).loc[targets[0], targets[1]]['Correlation']

            data = {'KL Divergence': [KLD],
                "Earth Mover's Distance": [EMD],
                'Correlation': [corr],
                'Selectivity': select,
                'Valency': [val]
            }
            df_temp = pd.DataFrame(data, columns=['KL Divergence', "Earth Mover's Distance", 'Correlation', 'Selectivity', 'Valency'])
            df = pd.concat([df, df_temp], ignore_index=True)
    sns.lineplot(data=df, x='KL Divergence', y='Selectivity', hue='Valency', ax=ax[0])
    sns.lineplot(data=df, x="Earth Mover's Distance", y='Selectivity', hue='Valency', ax=ax[1])
    sns.lineplot(data=df, x='Correlation', y='Selectivity', hue='Valency', ax=ax[2])
    ax[0].set(xscale='log')
    ax[1].set(xscale='log')

    return f
