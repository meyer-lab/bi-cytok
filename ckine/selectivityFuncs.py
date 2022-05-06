"""
Functions used in binding and selectivity analysis
"""
from .imports import importCITE, importReceptors
from .MBmodel import cytBindingModel_CITEseq, cytBindingModel_bispecCITEseq
from os.path import dirname, join
from scipy.optimize import minimize, Bounds
import pandas as pd
import seaborn as sns
import numpy as np

path_here = dirname(dirname(__file__))

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
            targetBound += cytBindingModel_bispecCITEseq(counts, betaAffs, 9.0, val, mut)

        for cellT in offTCells:
            for i, cd25Count in enumerate(cd25DF[cellT].item()):
                cd122Count = cd122DF[cellT].item()[i]
                epitopeCount = epitopeDF[cellT].item()[i]
                counts = [cd25Count, cd122Count, epitopeCount]
                offTargetBound += cytBindingModel_bispecCITEseq(counts, betaAffs, 9.0, val, mut)

    else:
        for i, cd25Count in enumerate(cd25DF[targCell].item()):
            cd122Count = cd122DF[targCell].item()[i]
            counts = [cd25Count, cd122Count]
            targetBound += cytBindingModel_CITEseq(counts, betaAffs, val, mut)

        for cellT in offTCells:
            for i, cd25Count in enumerate(cd25DF[cellT].item()):
                cd122Count = cd122DF[cellT].item()[i]
                counts = [cd25Count, cd122Count]
                offTargetBound += cytBindingModel_CITEseq(counts, betaAffs, val, mut)

    return targetBound, offTargetBound

def minSelecFunc(x, selectedDF, targCell, offTCells, epitope):
    """Provides the function to be minimized to get optimal selectivity"""
    targetBound = 0
    offTargetBound = 0

    recXaff = x

    epitopeDF = selectedDF.loc[(selectedDF.Type == 'Epitope')]
    cd25DF = selectedDF.loc[(selectedDF.Type == 'Standard') & (selectedDF.Epitope == 'CD25')]
    cd122DF = selectedDF.loc[(selectedDF.Type == 'Standard') & (selectedDF.Epitope == 'CD122')]

    for i, epCount in enumerate(epitopeDF[targCell].item()):
        cd25Count = cd25DF[targCell].item()[i]
        cd122Count = cd122DF[targCell].item()[i]
        counts = [cd25Count, cd122Count, epCount]
        targetBound += cytBindingModel_bispecCITEseq(counts, recXaff)
    for cellT in offTCells:
        for i, epCount in enumerate(epitopeDF[cellT].item()):
            cd25Count = cd25DF[cellT].item()[i]
            cd122Count = cd122DF[cellT].item()[i]
            counts = [cd25Count, cd122Count, epCount]

            offTargetBound += cytBindingModel_bispecCITEseq(counts, recXaff)

    return (offTargetBound) / (targetBound)

def optimizeDesign(targCell, offTcells, selectedDF, epitope):
    """ A more general purpose optimizer """
    if targCell == "NK":
        X0 = [6.0, 8]
    else:
        X0 = [7.0]

    optBnds = Bounds(np.full_like(X0, 6.0), np.full_like(X0, 9.0))

    optimized = minimize(minSelecFunc, X0, bounds=optBnds, args=(selectedDF, targCell, offTcells, epitope), jac="3-point")
    optSelectivity = optimized.fun[0]

    return optSelectivity


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
