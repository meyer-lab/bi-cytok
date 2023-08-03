from os.path import dirname, join
from .common import getSetup
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from ..selectivityFuncs import get_cell_bindings, getSampleAbundances, get_rec_vecs, optimizeDesign, minSelecFunc
from ..imports import importCITE
from pandas.api.types import CategoricalDtype

path_here = dirname(dirname(__file__))

def makeFigure():
    ax, f = getSetup((10, 8), (3, 2))
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
    cells = np.array(['CD8 Naive', 'NK', 'CD8 TEM', 'CD4 Naive', 'CD4 CTL', 'CD8 TCM', 'CD8 Proliferating',
    'Treg', 'CD4 TEM', 'NK Proliferating', 'NK_CD56bright'])
    offTCells = cells[cells != targCell]

    # This is just grabbing vectors of receptors to use in the function. Take a look at the output to see what's happening
    doseVec = np.logspace(-3, 3, num=20) #changed num = 2 so runs faster when practicing 
    doseVec = np.array(doseVec)
    epitopesDF = getSampleAbundances(epitopes, cells, "CellType2")
    prevOptAffs = [8.0, 8.0, 8.0]
    selectivity_values1 = []
    affinity_values1 = []
    selectivity_values2 = []
    affinity_values2 = []
    
    for dose in doseVec:
        #this function gets the optimal affinties and returns the optimal selectivity (first output), and affinities (second outputs) plot these dose on bottom for all, 2 have selectivity, 2 have affinity 
        optParams1 = optimizeDesign(secondary, epitope, targCell, offTCells, epitopesDF, dose, valency1, prevOptAffs)
        selectivity1 = optParams1[0]
        affinity1 = optParams1[1][0] #changed this added the [0]
        selectivity_values1.append(selectivity1)
        affinity_values1.append(affinity1)
    for dose in doseVec:
        #this function gets the optimal affinties and returns the optimal selectivity (first output), and affinities (second outputs) plot these dose on bottom for all, 2 have selectivity, 2 have affinity 
        optParams2 = optimizeDesign(secondary, epitope, targCell, offTCells, epitopesDF, dose, valency2, prevOptAffs)
        selectivity2 = optParams2[0]
        affinity2 = optParams2[1][0] # added the [0]
        selectivity_values2.append(selectivity2)
        affinity_values2.append(affinity2)
    
    # convert to np maybe? works.
    affinity_values1 = np.array(affinity_values1)
    affinity_values2 = np.array(affinity_values2)
    
    # check lengths
    print(len(doseVec))
    print(len(selectivity_values1))
    print(len(selectivity_values2))
    print(len(affinity_values1))
    print(len(affinity_values2))

    # Plot the optimal affinity and optimal selectivity you get at each dose
    sns.lineplot(x=doseVec, y=selectivity_values1, ax=ax[0], label='Selectivity for bivalent')
    ax[0].set_title('Valency 1 Selectivity')
    ax[0].set(xlabel='Dose', ylabel='Value')

    sns.lineplot(x=doseVec, y=np.ravel(affinity_values1), ax=ax[1], label='Affinity for bivalent')    
    ax[1].set_title('Valency 1 Affinity')
    ax[1].set(xlabel='Dose', ylabel='Value')

    # Then do the same thing but for higher valency (valency = 4) - refers to bi or tet
    sns.lineplot(x=doseVec, y=selectivity_values2, ax=ax[2], label='Selectivity for tetravalent')
    ax[2].set_title('Valency 2 Selectivity')
    ax[2].set(xlabel='Dose', ylabel='Value')

    sns.lineplot(x=doseVec, y=np.ravel(affinity_values2), ax=ax[3], label='Affinity for tetravalent')
    ax[3].set_title('Valency 2 Affinity')
    ax[3].set(xlabel='Dose', ylabel='Value')
    return f
