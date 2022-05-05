"""
This creates Figure 6, plotting Treg to off target signaling for vaying IL2Rb affinity for different IL2 formats
"""
from email.mime import base
from os.path import dirname, join

from .figureCommon import getSetup
from ..imports import importCITE, importReceptors
from ..MBmodel import polyc, getKxStar
import pandas as pd
import seaborn as sns
import numpy as np

path_here = dirname(dirname(__file__))

# Parameters: Off Target Cells, Target Cell(s), epitopes of interest, sample size


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

def getSignaling(betaAffs, targCell, offTCells, epitopesDF):
    """Returns total signaling summed over single cells for given parameters, can be adjusted for various purposes"""

    treg_sigs = np.zeros((8, betaAffs.size)) # 8 is used here because we are comparing 8 total signal types
    offTarg_sigs = np.zeros((8, betaAffs.size))

    # 0-2 IL2 WT
    # 3-5 R38Q
    # 6-7 Live/Dead
    muts = ['IL2', 'R38Q/H16N']
    vals = [1, 2, 4]

    for i, aff in enumerate(betaAffs):
        print(aff)
        for j, mut in enumerate(muts):
            for k, val in enumerate(vals):
                n = (3 * j) + k
                treg_sig, offTarg_sig = bindingCalc(epitopesDF, targCell, offTCells, aff, val, mut)
                treg_sigs[n, i] = treg_sig
                offTarg_sigs[n, i] = offTarg_sig

        treg_sig_bi, offTarg_sig_bi = bindingCalc(epitopesDF, targCell, offTCells, aff, 1, 'R38Q/H16N', bispec=True, epitope='CD25')
        treg_sigs[6, i] = treg_sig_bi
        offTarg_sigs[6, i] = offTarg_sig_bi

        treg_sig_bi, offTarg_sig_bi = bindingCalc(epitopesDF, targCell, offTCells, aff, 2, 'R38Q/H16N', bispec=True, epitope='CD25')
        treg_sigs[7, i] = treg_sig_bi
        offTarg_sigs[7, i] = offTarg_sig_bi

    return treg_sigs, offTarg_sigs



def getSampleAbundances(epitopes,cellList):
    """Given list of epitopes and cell types, returns a dataframe containing abundance data on a single cell level"""
    # This dataframe will later be filled with our epitope abundance by cells
    receptors = {'Epitope': epitopes}
    epitopesDF = pd.DataFrame(receptors)

    # Import CITE data
    CITE_DF = importCITE()

    # PRODUCING ERRORS
    # Get conv factors, average them to use on epitopes with unlisted conv facts
    # convFact = convFactCalc(ax[0])
    # meanConv = convFact.Weight.mean()


    #Sample sizes generated corresponding to cell list using mean values
    sampleSizes = []
    for cellType in cellList:
        cellSample = []
        for i in np.arange(10): # Averaging results of 10
            sampleDF = CITE_DF.sample(1000) # Of 1000 cells in the sample...
            sampleSize = int(len(sampleDF.loc[sampleDF["CellType2"] == cellType])) # ...How many are this cell type
            cellSample.append(sampleSize) # Sample size is equivalent to represented cell count out of 1000 cells
        meanSize = np.mean(cellSample)
        sampleSizes.append(int(meanSize))


    # For each  cellType in list
    for i, cellType in enumerate(cellList):
        # Generate sample size
        sampleSize = sampleSizes[i]
        # Create data frame of this size at random selection
        cellDF = CITE_DF.loc[CITE_DF["CellType2"] == cellType].sample(sampleSize)


        cellType_abdundances = [] 
        # For each epitope (being done on per cell basis)
        for e in epitopesDF.Epitope:
            # calculate abundance based on converstion factor
            if e == 'CD25':
                convFact = 77.136987 # The values are from convFactCalc
            elif e == 'CD122':
                convFact = 332.680090
            else:
                assert(False)
                #convFact = meanConv
                convFact = 100. #PLACEHOLDER DONT KEEP

            # Calculating abundance from cite data
            citeVal = cellDF[e].to_numpy() # Getting CITE signals for each cell
            abundance = citeVal * convFact # (CITE signal * conversion factor) = abundance
            cellType_abdundances.append(abundance) # Append abundances for individual cells into one list
            # add column with this name to epitopesDF and abundances list

        epitopesDF[cellType] = cellType_abdundances # This list will be located at Epitope x Cell Type in the DF

    return epitopesDF


def cytBindingModel(counts, betaAffs, val, mut, x=False, date=False):
    """Runs binding model for a given mutein, valency, dose, and cell type."""
    
    doseVec = np.array([0.1])
    recCount = np.ravel(counts)

    mutAffDF = pd.read_csv(join(path_here, "data/WTmutAffData.csv"))
    Affs = mutAffDF.loc[(mutAffDF.Mutein == mut)]

    Affs = np.power(np.array([Affs["IL2RaKD"].values, [betaAffs]]) / 1e9, -1)

    Affs = np.reshape(Affs, (1, -1))
    Affs = np.repeat(Affs, 2, axis=0)
    np.fill_diagonal(Affs, 1e2)  # Each cytokine can only bind one a and one b

    if doseVec.size == 1:
        doseVec = np.array([doseVec])
    output = np.zeros(doseVec.size)

    for i, dose in enumerate(doseVec):
        if x:
            output[i] = polyc(dose / 1e9, np.power(10, x[0]), recCount, [[val, val]], [1.0], Affs)[0][1]
        else:
            output[i] = polyc(dose / 1e9, getKxStar(), recCount, [[val, val]], [1.0], Affs)[0][1]

    return output


def cytBindingModel_bispec(counts, betaAffs, recXaff, val, mut, x=False):
    """Runs bispecific binding model for a given mutein, epitope, valency, dose, and cell type."""
  
    recXaff = np.power(10, recXaff)
    
    doseVec = np.array([0.1])
    recCount = np.ravel(counts)

    mutAffDF = pd.read_csv(join(path_here, "data/WTmutAffData.csv"))
    Affs = mutAffDF.loc[(mutAffDF.Mutein == mut)]

    Affs = np.power(np.array([Affs["IL2RaKD"].values, [betaAffs]]) / 1e9, -1)

    Affs = np.reshape(Affs, (1, -1))
    Affs = np.append(Affs, recXaff)
    holder = np.full((3, 3), 1e2)
    np.fill_diagonal(holder, Affs)
    Affs = holder

    if doseVec.size == 1:
        doseVec = np.array([doseVec])
    output = np.zeros(doseVec.size)

    for i, dose in enumerate(doseVec):
        if x:
            output[i] = polyc(dose / (val * 1e9), np.power(10, x[0]), recCount, [[val, val, val]], [1.0], Affs)[0][1]
        else:
            output[i] = polyc(dose / (val * 1e9), getKxStar(), recCount, [[val, val, val]], [1.0], Affs)[0][1]

    return output


def bindingCalc(df, targCell, offTCells, betaAffs, val,mut,bispec=False,epitope=None):
    """Calculates selectivity for no additional epitope"""
    targetBound = 0
    offTargetBound = 0

    cd25DF = df.loc[(df.Epitope == 'CD25')]
    cd122DF = df.loc[(df.Epitope == 'CD122')]

    if(bispec):
        epitopeDF = df.loc[(df.Epitope == epitope)]

        for i, cd25Count in enumerate(cd25DF[targCell].item()):
            cd122Count = cd122DF[targCell].item()[i]
            epitopeCount = epitopeDF[targCell].item()[i]
            counts = [cd25Count, cd122Count, epitopeCount]
            targetBound += cytBindingModel_bispec(counts, betaAffs, 9.0, val, mut)

        for cellT in offTCells:
            for i, cd25Count in enumerate(cd25DF[cellT].item()):
                cd122Count = cd122DF[cellT].item()[i]
                epitopeCount = epitopeDF[cellT].item()[i]
                counts = [cd25Count, cd122Count, epitopeCount]
                offTargetBound += cytBindingModel_bispec(counts, betaAffs, 9.0, val, mut)

    else:
        for i, cd25Count in enumerate(cd25DF[targCell].item()):
            cd122Count = cd122DF[targCell].item()[i]
            counts = [cd25Count, cd122Count]
            targetBound += cytBindingModel(counts, betaAffs, val, mut)

        for cellT in offTCells:
            for i, cd25Count in enumerate(cd25DF[cellT].item()):
                cd122Count = cd122DF[cellT].item()[i]
                counts = [cd25Count, cd122Count]
                offTargetBound += cytBindingModel(counts, betaAffs, val, mut)

    return targetBound, offTargetBound


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
