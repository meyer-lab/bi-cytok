"""
This creates Figure 5, used to find optimal epitope classifier.
"""
from os.path import dirname, join
from .common import getSetup
from ..selectivityFuncs import getSampleAbundances, optimizeDesign, selecCalc, get_rec_vecs, minSelecFunc
from ..imports import importCITE
from copy import copy
import pandas as pd
import seaborn as sns
import numpy as np


path_here = dirname(dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    ax, f = getSetup((9, 12), (1, 1))

    epitopesList = pd.read_csv(join(path_here, "data/epitopeList.csv"))
    epitopes = list(epitopesList['Epitope'].unique())  # List epitopes to be included in analysis

    # List cells to be included in analysis (Both on and off target)
    targCell = 'Treg'
    cells = importCITE()["CellType2"].unique()
    offTCells = cells[cells != targCell]
    #offTCells = ['CD8 Naive', 'NK', 'CD8 TEM', 'CD8 TCM']

    epitopesDF = getSampleAbundances(epitopes, cells)  # epitopesDF: Rows are eptitopes, columns are cell types.
    # Each frame contains a list of single cell abundances (of size determined in function) for that epitope and cell type
    # EpitopeDF now contains a data of single cell abundances for each cell type for each epitope
    epitopesDF["Selectivity"] = -1
    # New column which will hold selectivity per epitope
    targRecs, offTRecs = get_rec_vecs(epitopesDF, targCell, offTCells, "CD122", epitopes[0])
    targRecs[2, :] = 0
    offTRecs[2, :] = 0
    baseSelectivity = 1 / minSelecFunc([8, 8, 8], "CD122", "CD25", targRecs, offTRecs, 0.1, 2)
    print(baseSelectivity)

    for i, epitope in enumerate(epitopesDF['Epitope']):
        # New form
        optSelectivity = 1 / (optimizeDesign("CD122", "CD25", targCell, offTCells, epitopesDF, 1, 2, [8, 8, 8]))[0]
        epitopesDF.loc[epitopesDF['Epitope'] == epitope, 'Selectivity'] = optSelectivity  # Store selectivity in DF to be used for plots
        print(optSelectivity)


    # generate figures
    # bar plot of each epitope

    epitopesDF = epitopesDF.sort_values(by=['Selectivity'])
    xvalues = epitopesDF['Epitope']
    yvalues = ((epitopesDF['Selectivity'] / baseSelectivity) * 100) - 100
    print(yvalues)
    cmap = sns.color_palette("husl", 10)
    sns.barplot(x=xvalues, y=yvalues, palette=cmap, ax=ax[0]).set_title('Title')
    ax[0].set_ylabel("Selectivity (% increase over standard IL2)")

    return f
