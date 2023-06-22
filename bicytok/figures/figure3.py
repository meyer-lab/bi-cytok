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
    ax, f = getSetup((10, 8), (2, 2)) 

    # do this but for NK cells and the epitope (CD335) you found

    secondary = 'CD122'
    epitope = 'CD335'
    secondaryAff = 6.0
    valency1 = 2
    valency2 = 4

    # Epitoppes list lets the receptor function know what to grab - you can alter this to include CD335 - done

    epitopesList = pd.read_csv(join(path_here, "data/epitopeList.csv"))
    epitopes = list(epitopesList['Epitope'].unique())
    epitopes.append('CD335')

    # Change the target cell but this basically just tells us what we're optimizing for and against - done
    
    targCell = 'NK'
    cells = np.array(['Treg Memory', 'NK', 'Treg Naive', 'CD8 Naive', 'NK_2', 'CD8 TEM_1', 'CD4 Naive', 'CD4 CTL', 'CD4 TCM_3', 'CD4 TCM_2',
    'CD8 TCM_1', 'NK_4', 'Treg Naive', 'CD4 TEM_1', 'CD4 TEM_3', 'NK Proliferating', 'CD8 TEM_4', 'NK_CD56bright', 'CD4 TEM_4', 'NK_3'])
    offTCells = cells[cells != targCell]

    # This is just grabbing vectors of receptors to use in the function. Take a look at the output to see what's happening
    doseVec = np.logspace(-3, 3, num=20)
    epitopesDF = getSampleAbundances(epitopes, cells, "CellType3")
    targRecs, offTRecs = get_rec_vecs(epitopesDF, targCell, offTCells, secondary, epitope)

    prevOptAffs = [8.0, 8.0]
    selectivity_values1 = []
    affinity_values1 = []
    selectivity_values2 = []
    affinity_values2 = []
    
    for dose in doseVec:
        #this function gets the optimal affinties and returns the optimal selectivity (first output), and affinities (second outputs) plot these dose on bottom for all, 2 have selectivity, 2 have affinity 
        optParams1 = optimizeDesign(secondary, epitope, targCell, offTCells, epitopesDF, dose, valency1, prevOptAffs)
        selectivity1 = optParams1[0]
        affinity1 = optParams1[1]
        selectivity_values1.append(selectivity1)
        affinity_values1.append(affinity1)
    for dose in doseVec:
        #this function gets the optimal affinties and returns the optimal selectivity (first output), and affinities (second outputs) plot these dose on bottom for all, 2 have selectivity, 2 have affinity 
        optParams2 = optimizeDesign(secondary, epitope, targCell, offTCells, epitopesDF, dose, valency2, prevOptAffs)
        selectivity2 = optParams2[0]
        affinity2 = optParams2[1]
        selectivity_values2.append(selectivity2)
        affinity_values2.append(affinity2)
    
    # Plot the optimal affinity and optimal selectivity you get at each dose
    sns.lineplot(x=doseVec, y=selectivity_values1, ax=ax[0], label='Selectivity for bivalent')
    ax[0].set_title('Valency 1 Selectivity')
    
    sns.lineplot(x=doseVec, y=affinity_values1, ax=ax[1], label='Affinity for bivalent')
    ax[1].set_title('Valency 1 Affinity')

    # Then do the same thing but for higher valency (valency = 4) - refers to bi or tet
    sns.lineplot(x=doseVec, y=selectivity_values2, ax=ax[2], label='Selectivity for tetravalent')
    ax[2].set_title('Valency 2 Selectivity')

    sns.lineplot(x=doseVec, y=affinity_values2, ax=ax[3], label='Affinity for tetravalent')
    ax[3].set_title('Valency 2 Affinity')

    # Add labels and legend to the plot
    for ax in ax.flat:
        ax.set(xlabel='Dose', ylabel='Value')
        ax.legend()

    return f
