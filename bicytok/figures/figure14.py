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
    ax, f = getSetup((6, 3), (1, 2))

    signal = 'CD122'
    targets = ['CD25', 'CD278']
    valency = 2

    cells = np.array(['CD8 Naive', 'NK', 'Treg'])
    #cells = np.array(['CD8 Naive', 'NK', 'CD8 TEM', 'CD4 Naive', 'CD4 CTL', 'CD8 TCM', 'CD8 Proliferating',
    #    'Treg', 'CD4 TEM', 'NK Proliferating', 'NK_CD56bright'])
    targCell = 'Treg'
    offTCells = cells[cells != targCell]

    epitopesList = pd.read_csv(join(path_here, "data/epitopeList.csv"))
    epitopes = list(epitopesList['Epitope'].unique())
    epitopesDF = getSampleAbundances(epitopes, cells, "CellType2")

    doseVec = np.logspace(-3, 3, num=2)
    df = pd.DataFrame(columns=['Dose', 'Selectivity', 'Target Bound', 'Ligand'])
    targRecs, offTRecs = get_rec_vecs(epitopesDF, targCell, offTCells, signal, targets)

    prevOptAffs = [8.0, 8.0, 8.0]

    for _, dose in enumerate(doseVec):
        optParams = optimizeDesign(signal, targets, targCell, offTCells, epitopesDF, dose, valency, prevOptAffs)
        prevOptAffs = [optParams[1][0], optParams[1][1], optParams[1][2]]
        epitopeAff = optParams[1][2]

        data = {'Dose': [dose],
            'Selectivity': 1 / optParams[0],
            'Target Bound': optParams[2],
            'Ligand': "CD25+CD278"
        }

        df_temp = pd.DataFrame(data, columns=['Dose', 'Selectivity', 'Target Bound', 'Ligand'])
        df = df.append(df_temp, ignore_index=True)

    sns.lineplot(data=df, x='Dose', y='Selectivity', hue='Ligand', ax=ax[0])
    sns.lineplot(data=df, x='Dose', y='Target Bound', hue='Ligand', ax=ax[1])

    ax[0].set(xscale='log')
    ax[1].set(xscale='log')

    return f
