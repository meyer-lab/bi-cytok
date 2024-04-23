"""
Functions used in binding and selectivity analysis
"""
from .imports import importCITE, importReceptors
from .MBmodel import cytBindingModel
from scipy.optimize import minimize, Bounds
import pandas as pd
import numpy as np


def getSampleAbundances(epitopes: list, cellList: list, numCells=1000, cellCat="CellType2"):
    """Given list of epitopes and cell types, returns a dataframe containing abundance data on a single cell level
    Args:
        epitopes: list of epitopes for which you want abundance values
        cellList: list of cell types for which you want epitope abundance

    Returns:
        epitopesDF: dataframe containing single cell abundances of epitopes(rows) for each cell type(columns).
        Each frame contains a list of size corresponding to representative sample of cell type
    """

    # Import CITE data and drop unnecessary epitopes and cell types
    CITE_DF = importCITE()
    CITE_DF = CITE_DF[epitopes + [cellCat]]
    CITE_DF = CITE_DF.loc[CITE_DF[cellCat].isin(cellList)]

    # Get conv factors, average them to use on epitopes with unlisted conv facts
    meanConv = convFactCalc().Weight.mean()
    convFactDict = {
        'CD25': 77.136987,
        'CD122': 332.680090,
        'CD127': 594.379215
    }

    # Sample df generated
    sampleDF = CITE_DF.sample(numCells, random_state=42)

    # NOTE: Probably a better way to do this without a for loop
    for epitope in epitopes:
        sampleDF[epitope] = sampleDF[epitope].multiply(convFactDict.get(epitope, meanConv))

    return sampleDF

bispecOpt_Vec = np.vectorize(cytBindingModel, excluded=['holder', 'vals'], signature='(n),()->()')

def minSelecFunc(recXaffs: np.array, signal: str, targets: list, targRecs: np.array, offTRecs: np.array, dose: float, vals: list):
    """Serves as the function which will have its return value minimized to get optimal selectivity
    To be used in conjunction with optimizeDesign()
    Args:
        recXaff: receptor affinity which is modulated in optimize design

    Return:
        selectivity: value will be minimized, defined as ratio of off target to on target signaling
    """
    affs = pd.DataFrame()
    for i, recXaff in enumerate(recXaffs):
        affs = np.append(affs, np.power(10, recXaff))
    holder = np.full((recXaffs.size, recXaffs.size), 1e2)
    np.fill_diagonal(holder, affs)
    affs = holder

    targetBound = np.sum(bispecOpt_Vec(recCount=targRecs.to_numpy(), holder=affs, dose=dose, vals=vals))
    offTargetBound = np.sum(bispecOpt_Vec(recCount=offTRecs.to_numpy(), holder=affs, dose=dose, vals=vals))

    return offTargetBound / targetBound


def optimizeDesign(signal: str, targets: list, targCell: str, offTCells: list, selectedDF: pd.DataFrame, dose: float, valencies: list, prevOptAffs: list, cellCat="CellType2"):
    """ A general purzse optimizer used to minimize selectivity output by varying affinity parameter.
    Args:
        targCell: string cell type which is target and signaling is desired (basis of selectivity)
        offTCells: list of strings of cell types for which signaling is undesired
        selectedDf: contains epitope abundance information by cell type
        epitope: additional epitope to be targeted

    Return:
        optSelectivity: optimized selectivity value. Can also be modified to return optimized affinity parameter.
     """
    X0 = prevOptAffs
    minAffs = [7.0]
    maxAffs = [9.0]

    for target in targets:
        minAffs.append(7.0)
        maxAffs.append(9.0)

    optBnds = Bounds(np.full_like(X0, minAffs), np.full_like(X0, maxAffs))
    targRecs, offTRecs = get_rec_vecs(selectedDF, targCell, offTCells, signal, targets, cellCat)
    print('Optimize')
    optimized = minimize(minSelecFunc, X0, bounds=optBnds, args=(signal, targets, targRecs, offTRecs, dose, valencies), jac="3-point")
    print('Done')
    optSelectivity = optimized.fun
    optAffs = optimized.x

    return optSelectivity, optAffs


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


def convFactCalc() -> pd.DataFrame:
    """Returns conversion factors by marker for converting CITEseq signal into abundance"""
    CITE_DF = importCITE()
    cellToI = ["CD4 TCM", "CD8 Naive", "NK", "CD8 TEM", "CD4 Naive", "CD4 CTL", "CD8 TCM", "Treg", "CD4 TEM"]
    markers = ["CD122", "CD127", "CD25"]
    markerDF = None
    for marker in markers:
        for cell in cellToI:
            cellTDF = CITE_DF.loc[CITE_DF["CellType2"] == cell][marker]
            dftemp = pd.DataFrame({"Marker": [marker], "Cell Type": cell, "Amount": cellTDF.mean(), "Number": cellTDF.size})

            if markerDF is None:
                markerDF = dftemp
            else:
                markerDF = pd.concat([markerDF, dftemp])

    markerDF = markerDF.replace({"Marker": markDict, "Cell Type": cellDict})
    markerDFw = None
    for marker in markerDF.Marker.unique():
        for cell in markerDF["Cell Type"].unique():
            subDF = markerDF.loc[(markerDF["Cell Type"] == cell) & (markerDF["Marker"] == marker)]
            wAvg = np.sum(subDF.Amount.values * subDF.Number.values) / np.sum(subDF.Number.values)
            dftemp = pd.DataFrame({"Marker": [marker], "Cell Type": cell, "Average": wAvg})

            if markerDFw is None:
                markerDFw = dftemp
            else:
                markerDFw = pd.concat([markerDFw, dftemp])

    recDF = importReceptors()
    weightDF = None

    for rec in markerDFw.Marker.unique():
        CITEval = np.array([])
        Quantval = np.array([])
        for cell in markerDF["Cell Type"].unique():
            CITEval = np.concatenate((CITEval, markerDFw.loc[(markerDFw["Cell Type"] == cell) & (markerDFw["Marker"] == rec)].Average.values))
            Quantval = np.concatenate((Quantval, recDF.loc[(recDF["Cell Type"] == cell) & (recDF["Receptor"] == rec)].Mean.values))
        dftemp = pd.DataFrame({"Receptor": [rec], "Weight": np.linalg.lstsq(np.reshape(CITEval, (-1, 1)).astype(float), Quantval, rcond=None)[0]})

        if weightDF is None:
            weightDF = dftemp
        else:
            weightDF = pd.concat([weightDF, dftemp])

    return weightDF


def get_rec_vecs(df: pd.DataFrame, targCell: str, offTCells: list, signal: str, targets: list, cellCat: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns vector of target and off target receptors"""
    dfTargCell = df.loc[df[cellCat] == targCell]
    countTarg = dfTargCell[[signal] + targets]

    dfOffTCell = df.loc[df[cellCat].isin(offTCells)]
    countOffT = dfOffTCell[[signal] + targets]

    return countTarg, countOffT


def get_cell_bindings(recXaffs: np.ndarray, cells: list, df: pd.DataFrame, secondary: str, epitope: str, dose: float, valency: np.ndarray):
    df_return = pd.DataFrame(columns=['Cell Type', 'Secondary Bound', 'Total Secondary'])
    
    affs = pd.DataFrame()
    for i, recXaff in enumerate(recXaffs):
        affs = np.append(affs, np.power(10, recXaff))
    holder = np.full((recXaffs.size, recXaffs.size), 1e2)
    np.fill_diagonal(holder, affs)
    affs = holder

    cd25DF = df.loc[(df.Epitope == 'CD25')]
    secondaryDF = df.loc[(df.Epitope == secondary)]
    if epitope != None:
        df2 = df.loc[(df.Epitope == epitope)]
    else:
        df2 = df.loc[(df.Epitope == 'CD25')]

    for cell in cells:
        numCells = df2[cell].item().size

        cd25CountTarg = np.zeros(numCells)
        secondaryCountTarg = np.zeros(numCells)
        epCountvecTarg = np.zeros(numCells)
        for i, epCount in enumerate(df2[cell].item()):
            cd25CountTarg[i] = cd25DF[cell].item()[i]
            secondaryCountTarg[i] = secondaryDF[cell].item()[i]
            epCountvecTarg[i] = epCount

        recs = np.array([cd25CountTarg, secondaryCountTarg, epCountvecTarg])

        secondaryBound = 0.0
        for i in range(recs.shape[1]):
            secondaryBound += cytBindingModel(recs[:, i], affs[0:3], dose, valency)

        data = {'Cell Type': [cell],
            'Secondary Bound': [secondaryBound / numCells][0],
            'Total Secondary': [np.sum(recs[1]) / numCells]
        }

        df_temp = pd.DataFrame(data, columns=['Cell Type', 'Secondary Bound', 'Total Secondary'])
        df_return = pd.concat((df_return, df_temp))

    return df_return

