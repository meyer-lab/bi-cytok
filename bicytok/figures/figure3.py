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

    # do this but for NK cells and the epitope you found

    secondary = 'CD122'
    epitope = 'CD278'
    secondaryAff = 6.0
    valency = 2

    wtIL2RaAff = 8.222
    wtSecondaryAff = 7.65072247
    wtEpitopeAff = 9.14874165

    cells1 = np.array(['Treg', 'CD8 Naive', 'NK', 'CD8 TEM', 'CD4 Naive', 'CD4 CTL', 'CD8 TCM', 'CD4 TEM', 'NK Proliferating',
    'NK_CD56bright'])
    cells2 = np.array(['Treg Memory', 'Treg Naive'])

    # Epitoppes list lets the receptor function know what to grab - you can alter this to include CD335

    epitopesList = pd.read_csv(join(path_here, "data/epitopeList.csv"))
    epitopes = list(epitopesList['Epitope'].unique())

    # Change the target cell but this basically just tells us what we're optimizing for and against
    
    targCell = 'Treg Memory'
    cells = np.array(['Treg Memory', 'Treg Naive', 'CD8 Naive', 'NK_2', 'CD8 TEM_1', 'CD4 Naive', 'CD4 CTL', 'CD4 TCM_3', 'CD4 TCM_2',
    'CD8 TCM_1', 'NK_4', 'Treg Naive', 'CD4 TEM_1', 'CD4 TEM_3', 'NK Proliferating', 'CD8 TEM_4', 'NK_CD56bright', 'CD4 TEM_4', 'NK_3'])
    offTCells = cells[cells != targCell]

    # This is just grabbing vectors of receptors to use in the function. Take a look at the output to see what's happening
    doseVec = np.logspace(-3, 3, num=20)
    epitopesDF = getSampleAbundances(epitopes, cells, "CellType3")
    targRecs, offTRecs = get_rec_vecs(epitopesDF, targCell, offTCells, secondary, epitope)

    prevOptAffs = [8.0, 8.0, 8.0]

    for dose in doseVec:
        #this function gets the optimal affinties and returns the optimal selectivity (first output), and affinities (second outputs)
        optParams = optimizeDesign(secondary, epitope, targCell, offTCells, epitopesDF, 1e-9, valency, prevOptAffs)

    # Plot the optimal affinity and optimal selectivity you get at each dose

    # Then do the same thing but for higher valency (valency = 4)

    return f
