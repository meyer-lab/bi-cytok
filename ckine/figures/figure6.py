"""
This creates Figure 6, plotting Treg to off target signaling for vaying IL2Rb affinity for different IL2 formats
"""
from email.mime import base
from os.path import dirname, join

from .figureCommon import getSetup
from ..imports import importCITE, importReceptors
from ..selectivityFuncs import getSampleAbundances, getSignaling
import pandas as pd
import seaborn as sns 
import numpy as np

path_here = dirname(dirname(__file__))

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    ax, f = getSetup((13, 4), (1, 3))

    # List epitopes to be included in analysis
    epitopes = ['CD25','CD122']
    # List cells to be included in analysis (Both on and off target)
    targCell = 'Treg'
    offTCells = ['CD8 Naive', 'NK', 'CD8 TEM', 'CD8 TCM']
    cells = offTCells + [targCell]

    epitopesDF = getSampleAbundances(epitopes,cells)  # epitopesDF: Rows are eptitopes, columns are cell types.
    # Each frame contains a list of single cell abundances (of size determined in function) for that epitope and cell type
    
    #range from 0.01 <-> 100
    betaAffs = np.logspace(-4, 2, 3) #2s should be 40s
    # Fills arrays of target and off target signals for given array of parameters
    treg_sigs, offTarg_sigs = getSignaling(betaAffs, targCell, offTCells, epitopesDF)

    # print(y_ticks)
    def plotSignals(types, ax):
        # Add standard colors/line types
        if 'WT' in types:
            ax.plot(norm(treg_sigs[0]), norm(offTarg_sigs[0]), label='WT', c='blue')
            ax.plot(norm(treg_sigs[1]), norm(offTarg_sigs[1]), label='WT Bival', c='green')
            ax.plot(norm(treg_sigs[2]), norm(offTarg_sigs[2]), label='WT Tetraval', c='c')
        if 'R38Q/H16N' in types:
            ax.plot(norm(treg_sigs[3]), norm(offTarg_sigs[3]), '--', label='R38Q/H16N', c='red')
            ax.plot(norm(treg_sigs[4]), norm(offTarg_sigs[4]), '--', label='R38Q/H16N Bival', c='y')
            ax.plot(norm(treg_sigs[5]), norm(offTarg_sigs[5]), '--', label='R38Q/H16N Tetraval', c='orange')
        if 'Live/Dead' in types:
            ax.plot(norm(treg_sigs[6]), norm(offTarg_sigs[6]), '-.', label='CD25 Live/Dead', c='indigo')
            ax.plot(norm(treg_sigs[7]), norm(offTarg_sigs[7]), '-.', label='CD25 Bivalent Live/Dead', c='magenta')

        ax.set_xlabel('Treg Signaling', fontsize=12)
        ax.set_ylabel('Off Target Signaling', fontsize=12)
        ax.legend()

    plotSignals(['WT', 'R38Q/H16N'], ax[0])
    plotSignals(['WT', 'Live/Dead'], ax[1])
    plotSignals(['R38Q/H16N', 'Live/Dead'], ax[2])
    f.suptitle('Treg vs. Off Target Signaling Varing Beta Affinity', fontsize=18)

    return f

# Normalizes data to 1
def norm(data):
    return data / max(data)


cellDict = {"CD4 Naive": "Thelper",
            "CD4 CTL": "Thelper",
            "CD4 TCM": "Thelper",
            "CD4 TEM": "Thelper",
            "NK": "NK",
            "CD8 Naive": "CD8",
            "CD8 TCM": "CD8",
            "CD8 TEM": "CD8",
            "Treg": "Treg"}


markDict = {"CD25": "IL2Ra",
            "CD122": "IL2Rb",
            "CD127": "IL7Ra",
            "CD132": "gc"}


def convFactCalc(ax):
    """Fits a ridge classifier to the CITE data and plots those most highly correlated with T reg"""
    CITE_DF = importCITE()
    cellToI = ["CD4 TCM", "CD8 Naive", "NK", "CD8 TEM", "CD4 Naive", "CD4 CTL", "CD8 TCM", "Treg", "CD4 TEM"]
    markers = ["CD122", "CD127", "CD25"]
    markerDF = pd.DataFrame(columns=["Marker", "Cell Type", "Amount", "Number"])
    for marker in markers:
        for cell in cellToI:
            cellTDF = CITE_DF.loc[CITE_DF["CellType2"] == cell][marker]
            markerDF = markerDF.append(pd.DataFrame({"Marker": [marker], "Cell Type": cell, "Amount": cellTDF.mean(), "Number": cellTDF.size}))

    markerDF = markerDF.replace({"Marker": markDict, "Cell Type": cellDict})
    markerDFw = pd.DataFrame(columns=["Marker", "Cell Type", "Average"])
    for marker in markerDF.Marker.unique():
        for cell in markerDF["Cell Type"].unique():
            subDF = markerDF.loc[(markerDF["Cell Type"] == cell) & (markerDF["Marker"] == marker)]
            wAvg = np.sum(subDF.Amount.values * subDF.Number.values) / np.sum(subDF.Number.values)
            markerDFw = markerDFw.append(pd.DataFrame({"Marker": [marker], "Cell Type": cell, "Average": wAvg}))

    recDF = importReceptors()
    weightDF = pd.DataFrame(columns=["Receptor", "Weight"])

    for rec in markerDFw.Marker.unique():
        CITEval = np.array([])
        Quantval = np.array([])
        for cell in markerDF["Cell Type"].unique():
            CITEval = np.concatenate((CITEval, markerDFw.loc[(markerDFw["Cell Type"] == cell) & (markerDFw["Marker"] == rec)].Average.values))
            Quantval = np.concatenate((Quantval, recDF.loc[(recDF["Cell Type"] == cell) & (recDF["Receptor"] == rec)].Mean.values))
            CITEval = np.reshape(CITEval, (-1, 1))
            CITEval = CITEval.astype(float)
        weightDF = weightDF.append(pd.DataFrame({"Receptor": [rec], "Weight": np.linalg.lstsq(CITEval, Quantval, rcond=None)[0]}))
        print("Success")
    return weightDF
