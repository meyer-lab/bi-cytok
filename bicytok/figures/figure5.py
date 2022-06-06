"""
This creates Figure 5, used to find optimal epitope classifier.
"""
from os.path import dirname, join
from .common import getSetup
from ..imports import importCITE
from valentbind import polyc
from ..selectivityFuncs import getSampleAbundances, optimizeDesign, selecCalc
from copy import copy
import pandas as pd
import seaborn as sns
import numpy as np


path_here = dirname(dirname(__file__))


# Later can we test by making a fake cell with 0 or -1 for every value and verifying positions are maintained?
def makeFigure():
    """Get a list of the axis objects and create a figure"""
    ax, f = getSetup((9, 12), (1, 1))

    epitopesDF = pd.DataFrame(columns={"Classifier", "Epitope", "Selectivity"})
    epitopesDF = pd.read_csv(join(path_here, "data/epitopeList.csv"))
    epitopes = list(epitopesDF['Epitope'].unique())  # List epitopes to be included in analysis

    # List cells to be included in analysis (Both on and off target)
    targCell = 'Treg'
    offTCells = ['CD8 Naive', 'NK', 'CD8 TEM', 'CD8 TCM']
    cells = offTCells + [targCell]

    epitopesDF = getSampleAbundances(epitopes, cells)  # epitopesDF: Rows are eptitopes, columns are cell types.
    # Each frame contains a list of single cell abundances (of size determined in function) for that epitope and cell type

    # EpitopeDF now contains a data of single cell abundances for each cell type for each epitope
    epitopesDF["Selectivity"] = -1
    # New column which will hold selectivity per epitope

    for epitope in epitopesDF['Epitope']:

        # New form
        optSelectivity = 1 / (optimizeDesign(targCell, offTCells, epitopesDF, epitope))
        epitopesDF.loc[epitopesDF['Epitope'] == epitope, 'Selectivity'] = optSelectivity  # Store selectivity in DF to be used for plots

    baseSelectivity = 1 / (selecCalc(epitopesDF, targCell, offTCells))

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
