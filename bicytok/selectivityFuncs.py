"""
Functions used in binding and selectivity analysis
"""

from .imports import importCITE, importReceptors
from .MBmodel import cytBindingModel
from scipy.optimize import minimize, Bounds
import pandas as pd
import numpy as np


def getSampleAbundances(
    epitopes: list, cellList: list, numCells=1000, cellCat="CellType2"
):
    """Given list of epitopes and cell types, returns a dataframe containing receptor abundance data on a single-cell level.
    Args:
        epitopes: list of epitopes for which you want abundance values
        cellList: list of cell types for which you want epitope abundance
        numCells: number of cells to sample from for abundance calculations, default to sampling from 1000 cells
        cellCat: cell type categorization level, see cell types/subsets in CITE data
    Return:
        sampleDF: dataframe containing single cell abundances of receptors (column) for each individual cell (row),
        with final column being cell type from the cell type categorization level set by cellCat
    """

    # Import CITE data and drop unnecessary epitopes and cell types
    CITE_DF = importCITE()
    CITE_DF_new = CITE_DF[epitopes + [cellCat]]
    CITE_DF_new = CITE_DF_new.loc[CITE_DF_new[cellCat].isin(cellList)]

    # Get conv factors, average them to use on epitopes with unlisted conv facts
    meanConv = convFactCalc(CITE_DF).Weight.mean()
    # convFactDict values calculated by convFactCalc
    convFactDict = {"CD25": 77.136987, "CD122": 332.680090, "CD127": 594.379215}

    # Sample df generated
    sampleDF = CITE_DF_new.sample(numCells, random_state=42)

    for epitope in epitopes:
        sampleDF[epitope] = sampleDF[epitope].multiply(
            convFactDict.get(epitope, meanConv)
        )

    return sampleDF

# Vectorization function for cytBindingModel
bispecOpt_Vec = np.vectorize(
    cytBindingModel, excluded=["recXaffs", "vals"], signature="(n),()->()"
)


def minSelecFunc(
    recXaffs: np.ndarray,
    signal: str,
    targets: list,
    targRecs: pd.DataFrame,
    offTRecs: pd.DataFrame,
    dose: float,
    vals: np.ndarray,
):
    """Serves as the function which will have its return value minimized to get optimal selectivity
    To be used in conjunction with optimizeDesign()
    Args:
        recXaff: receptor affinity which is modulated in optimize design
        signal: signaling receptor
        targets: list of targeted receptors
        targRecs: dataframe of receptors counts of target cell type
        offTRecs: dataframe of receptors counts of off-target cell types
        dose: ligand concentration/dose that is being modeled
        vals: array of valencies of each ligand epitope
    Return:
        selectivity: value will be minimized, defined as ratio of off target to on target signaling
    """

    affs = get_affs(recXaffs)

    targetBound = (
        np.sum(
            bispecOpt_Vec(
                recCount=targRecs.to_numpy(), recXaffs=affs, dose=dose, vals=vals
            )
        )
        / targRecs.shape[0]
    )
    offTargetBound = (
        np.sum(
            bispecOpt_Vec(
                recCount=offTRecs.to_numpy(), recXaffs=affs, dose=dose, vals=vals
            )
        )
        / offTRecs.shape[0]
    )

    return offTargetBound / targetBound


def optimizeDesign(
    signal: str,
    targets: list,
    targCell: str,
    offTCells: list,
    selectedDF: pd.DataFrame,
    dose: float,
    valencies: np.ndarray,
    prevOptAffs: list,
    cellCat="CellType2",
):
    """A general purzse optimizer used to minimize selectivity output by varying affinity parameter.
    Args:
        targCell: string cell type which is target and signaling is desired (basis of selectivity)
        offTCells: list of strings of cell types for which signaling is undesired
        selectedDf: contains epitope abundance information by cell type
        epitope: additional epitope to be targeted

    Return:
        optSelectivity: optimized selectivity value. Can also be modified to return optimized affinity parameter.
    """
    X0 = prevOptAffs
    minAffs = [7.0] * (len(targets) + 1)
    maxAffs = [9.0] * (len(targets) + 1)

    optBnds = Bounds(np.full_like(X0, minAffs), np.full_like(X0, maxAffs))
    targRecs, offTRecs = get_rec_vecs(
        selectedDF, targCell, offTCells, signal, targets, cellCat
    )
    print("Optimize")
    optimized = minimize(
        minSelecFunc,
        X0,
        bounds=optBnds,
        args=(
            signal,
            targets,
            targRecs.drop([cellCat], axis=1),
            offTRecs.drop([cellCat], axis=1),
            dose,
            valencies,
        ),
        jac="3-point",
    )
    print("Done")
    optSelectivity = optimized.fun
    optAffs = optimized.x

    return optSelectivity, optAffs


cellDict = {
    "CD4 Naive": "Thelper",
    "CD4 CTL": "Thelper",
    "CD4 TCM": "Thelper",
    "CD4 TEM": "Thelper",
    "NK": "NK",
    "CD8 Naive": "CD8",
    "CD8 TCM": "CD8",
    "CD8 TEM": "CD8",
    "Treg": "Treg",
}


markDict = {"CD25": "IL2Ra", "CD122": "IL2Rb", "CD127": "IL7Ra", "CD132": "gc"}


# NOTE: Come back to this later
def convFactCalc(CITE_DF: pd.DataFrame) -> pd.DataFrame:
    """Returns conversion factors by marker for converting CITEseq signal into abundance"""
    cellToI = [
        "CD4 TCM",
        "CD8 Naive",
        "NK",
        "CD8 TEM",
        "CD4 Naive",
        "CD4 CTL",
        "CD8 TCM",
        "Treg",
        "CD4 TEM",
    ]
    markers = ["CD122", "CD127", "CD25"]
    markerDF = pd.DataFrame()
    for marker in markers:
        for cell in cellToI:
            cellTDF = CITE_DF.loc[CITE_DF["CellType2"] == cell][marker]
            dftemp = pd.DataFrame(
                {
                    "Marker": [marker],
                    "Cell Type": cell,
                    "Amount": cellTDF.mean(),
                    "Number": cellTDF.size,
                }
            )

            if markerDF is pd.DataFrame():
                markerDF = dftemp
            else:
                markerDF = pd.concat([markerDF, dftemp])

    markerDF = markerDF.replace({"Marker": markDict, "Cell Type": cellDict})
    markerDFw = pd.DataFrame()
    for marker in markerDF.Marker.unique():
        for cell in markerDF["Cell Type"].unique():
            subDF = markerDF.loc[
                (markerDF["Cell Type"] == cell) & (markerDF["Marker"] == marker)
            ]
            wAvg = np.sum(subDF.Amount.values * subDF.Number.values) / np.sum(
                subDF.Number.values
            )
            dftemp = pd.DataFrame(
                {"Marker": [marker], "Cell Type": cell, "Average": wAvg}
            )

            if markerDFw is pd.DataFrame():
                markerDFw = dftemp
            else:
                markerDFw = pd.concat([markerDFw, dftemp])

    recDF = importReceptors()
    weightDF = None

    for rec in markerDFw.Marker.unique():
        CITEval = np.array([])
        Quantval = np.array([])
        for cell in markerDF["Cell Type"].unique():
            CITEval = np.concatenate(
                (
                    CITEval,
                    markerDFw.loc[
                        (markerDFw["Cell Type"] == cell) & (markerDFw["Marker"] == rec)
                    ].Average.values,
                )
            )
            Quantval = np.concatenate(
                (
                    Quantval,
                    recDF.loc[
                        (recDF["Cell Type"] == cell) & (recDF["Receptor"] == rec)
                    ].Mean.values,
                )
            )
        dftemp = pd.DataFrame(
            {
                "Receptor": [rec],
                "Weight": np.linalg.lstsq(
                    np.reshape(CITEval, (-1, 1)).astype(float), Quantval, rcond=None
                )[0],
            }
        )

        if weightDF is None:
            weightDF = dftemp
        else:
            weightDF = pd.concat([weightDF, dftemp])

    return weightDF


def get_rec_vecs(
    df: pd.DataFrame,
    targCell: str,
    offTCells: list,
    signal: str,
    targets: list,
    cellCat="CellType2",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns vector of target and off target receptors"""
    dfTargCell = df.loc[df[cellCat] == targCell]
    countTarg = dfTargCell[[signal] + targets + [cellCat]]

    dfOffTCell = df.loc[df[cellCat].isin(offTCells)]
    countOffT = dfOffTCell[[signal] + targets + [cellCat]]

    return countTarg, countOffT


def get_cell_bindings(
    df: np.ndarray,
    signal: str,
    targets: list,
    recXaffs: np.ndarray,
    dose: float,
    vals: np.ndarray,
    cellCat="CellType2",
):
    targRecs = pd.DataFrame()
    df_return = pd.DataFrame()

    targRecs = df[[signal] + targets]
    affs = get_affs(recXaffs)

    targBound = bispecOpt_Vec(
        recCount=targRecs.to_numpy(), recXaffs=affs, dose=dose, vals=vals
    )

    df_return = df[[signal] + [cellCat]]
    df_return.insert(0, "Receptor Bound", targBound, True)
    df_return = df_return.groupby([cellCat]).mean(0)

    return df_return


def get_affs(recXaffs: np.ndarray):
    affs = pd.DataFrame()
    for i, recXaff in enumerate(recXaffs):
        affs = np.append(affs, np.power(10, recXaff))
    holder = np.full((recXaffs.size, recXaffs.size), 1e2)
    np.fill_diagonal(holder, affs)
    affs = holder

    return affs
