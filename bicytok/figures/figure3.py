from .common import getSetup
import pandas as pd
import seaborn as sns
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from ..selectivityFuncs import getSampleAbundances, optimizeDesign


def makeFigure():
    ax, f = getSetup((10, 8), (3, 2))
    # do this but for NK cells and the epitope (CD335) you found

    secondary = 'CD122'
    epitope = 'CD335'
    valency1 = 2
    valency2 = 4

    # Epitoppes list lets the receptor function know what to grab - you can alter this to include CD335 - done

    epitopesList = pd.read_csv("./bicytok/data/epitopeList.csv")
    epitopes = list(epitopesList['Epitope'].unique())
    epitopes.append('CD335')
    # Change the target cell but this basically just tells us what we're optimizing for and against - done
    
    targCell = 'NK'
    cells = ['CD8 Naive', 'NK', 'CD8 TEM', 'CD4 Naive', 'CD4 CTL', 'CD8 TCM', 'CD8 Proliferating',
    'Treg', 'CD4 TEM', 'NK Proliferating', 'NK_CD56bright']
    offTCells = [c for c in cells if c != targCell]

    # This is just grabbing vectors of receptors to use in the function. Take a look at the output to see what's happening
    doseVec = np.logspace(-3, 3, num=5) #changed num = 2 so runs faster when practicing 
    epitopesDF = getSampleAbundances(epitopes, cells, "CellType2")
    futures1 = []
    futures2 = []

    executor = ProcessPoolExecutor(max_workers=10)

    print("Starting optimization")

    # this function gets the optimal affinties and returns the optimal selectivity (first output),
    # and affinities (second outputs) plot these dose on bottom for all, 2 have selectivity, 2 have affinity
    for dose in doseVec:
        futures1.append(executor.submit(optimizeDesign, secondary, epitope, targCell, offTCells, epitopesDF, dose, valency1))

    for dose in doseVec:
        futures2.append(executor.submit(optimizeDesign, secondary, epitope, targCell, offTCells, epitopesDF, dose, valency2))

    affinity_values1 = np.array([fut.result()[1][0] for fut in futures1])
    affinity_values2 = np.array([fut.result()[1][0] for fut in futures2])
    selectivity_values1 = np.array([fut.result()[0] for fut in futures1])
    selectivity_values2 = np.array([fut.result()[0] for fut in futures2])

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
