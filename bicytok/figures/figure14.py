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
    ax, f = getSetup((12, 6), (1, 2))

    signal = ['CD122', 1]
    allTargets = [[('CD25', 1)], [('CD278', 1)], [('CD25', 1), ('CD278', 1)], [('CD278', 1), ('CD45RB', 1)],
        [('CD278', 1), ('CD81', 1)], [('CD278', 1), ('CD4-2', 1)], [('CD25', 1), ('CD278', 1), ('CD45RB', 1)],
        [('CD25', 1), ('CD278', 1), ('CD81', 1)], [('CD25', 1), ('CD278', 1), ('CD4-2', 1)],
        [('CD25', 4)], [('CD278', 4)], [('CD25', 4), ('CD278', 4)], [('CD278', 4), ('CD45RB', 4)],
        [('CD278', 4), ('CD81', 4)], [('CD278', 4), ('CD4-2', 4)], [('CD25', 4), ('CD278', 4), ('CD45RB', 4)],
        [('CD25', 4), ('CD278', 4), ('CD81', 4)], [('CD25', 4), ('CD278', 4), ('CD4-2', 4)]]

    """wt105Aff = 1.1e-1
    wt109Aff = 5.4e-2
    affs = [wt105Aff, wt109Aff]"""

    cells = np.array(['CD8 Naive', 'NK', 'CD8 TEM', 'CD4 Naive', 'CD4 CTL', 'CD8 TCM', 'CD8 Proliferating',
        'Treg', 'CD4 TEM', 'NK Proliferating', 'NK_CD56bright'])
    targCell = 'Treg'
    offTCells = cells[cells != targCell]

    epitopesList = pd.read_csv(join(path_here, "data/epitopeList.csv"))
    epitopes = list(epitopesList['Epitope'].unique())
    epitopesDF = getSampleAbundances(epitopes, cells)

    doseVec = np.logspace(-2, 2, num=20)
    df = pd.DataFrame(columns=['Dose', 'Selectivity', 'Target Bound', 'Ligand'])
    df2 = pd.DataFrame(columns=['Ligand', 'Dose', 'Affinities'])

    for targetPairs in allTargets:
        prevOptAffs = [8.0]
        valencies = [signal[1]]
        targets = []
        naming = []
        for target, valency in targetPairs:
            prevOptAffs.append(8.0)
            targets.append(target)
            valencies.append(valency)
            naming.append('{} ({})'.format(target, valency))

        #targRecs, offTRecs = get_rec_vecs(epitopesDF, targCell, offTCells, signal[0], targets)

        for _, dose in enumerate(doseVec):
            optParams = optimizeDesign(signal[0], targets, targCell, offTCells, epitopesDF, dose, valencies, prevOptAffs)
            prevOptAffs = optParams[1]
                
            data = {'Dose': [dose],
                'Selectivity': 1 / optParams[0],
                'Target Bound': optParams[2],
                'Ligand': ' + '.join(naming)
            }
            df_temp = pd.DataFrame(data, columns=['Dose', 'Selectivity', 'Target Bound', 'Ligand'])
            df = pd.concat([df, df_temp], ignore_index=True)
            print(optParams[1])

            data = {'Ligand': ' + '.join(naming),
                'Dose': dose,
                'Affinities': optParams[1]
            }
            df2_temp = pd.DataFrame(data, columns=['Ligand', 'Dose', 'Affinities'])
            df2 = pd.concat([df2, df2_temp], ignore_index=True)

        """# TEMP START
            LD = minSelecFunc(affs, signal[0], targets, targRecs, offTRecs, dose, valencies)
            data = {'Dose': [dose],
                'Selectivity': 1 / LD,
                'Target Bound': minSelecFunc.targetBound,
                'Ligand': "Live/Dead"
            }
            df_temp = pd.DataFrame(data, columns=['Dose', 'Selectivity', 'Target Bound', 'Ligand'])
            df = pd.concat([df, df_temp], ignore_index=True)
        # TEMP END"""
    
    print(df2)
            
    sns.lineplot(data=df, x='Dose', y='Selectivity', hue='Ligand', ax=ax[0])
    sns.lineplot(data=df, x='Dose', y='Target Bound', hue='Ligand', ax=ax[1])
    ax[0].set(xscale='log')
    ax[1].set(xscale='log')

    return f
