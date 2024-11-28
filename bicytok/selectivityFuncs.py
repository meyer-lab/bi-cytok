"""
Functions used in binding and selectivity analysis
"""

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, minimize

from bicytok.imports import importCITE, importReceptors
from bicytok.MBmodel import cytBindingModel


# Called in minSelecFunc and get_cell_bindings
def get_affs(recXaffs: np.ndarray) -> np.ndarray:
    """
    Structures array of receptor affinities to be compatible with the binding model
    Args:
        recXaffs: receptor affinities
    Return:
        affs: restructured receptor affinities
    """

    affs = pd.DataFrame()
    for i, recXaff in enumerate(recXaffs):
        affs = np.append(affs, np.power(10, recXaff))

    holder = np.full((recXaffs.size, recXaffs.size), 1e2)
    np.fill_diagonal(holder, affs)
    affs = holder

    return affs


# Called in optimizeDesign
def minSelecFunc(  # convert to an inner function of optimizeDesign???
    recXaffs: np.ndarray,
    targRecs: pd.DataFrame,
    offTargRecs: pd.DataFrame,
    dose: float,
    valencies: np.ndarray,
) -> float:
    """
    Serves as the function which will have its return value minimized to get
        optimal selectivity
        To be used in conjunction with optimizeDesign()
    Args:
        recXaff: receptor affinities which are modulated in optimize design
        signal: signaling receptor
        targets: list of targeted receptors
        targRecs: dataframe of receptors counts of target cell type
        offTargRecs: dataframe of receptors counts of off-target cell types
        dose: ligand concentration/dose that is being modeled
        valencies: array of valencies of each ligand epitope
    Return:
        selectivity: value will be minimized, defined as ratio of off target
            to on target signaling this is selectivity for the off target cells
            and is minimized to maximize selectivity for the target cell
    """

    affs = get_affs(recXaffs)

    cytBindingModel_vec = np.vectorize(  # Vectorization function for cytBindingModel
        cytBindingModel, excluded=["recXaffs", "valencies"], signature="(n),()->()"
    )

    targetBound = (
        np.sum(  # replace sum with flatten or reshape (Armaan's suggestion)
            cytBindingModel_vec(
                recCount=targRecs.to_numpy(),
                recXaffs=affs,
                dose=dose,
                valencies=valencies,
            )
        )
        / targRecs.shape[0]
        # Armaan: I believe that you can delete this denominator, as targRecs should
        # always have one row consisting of the one target cell. Alternatively
        # (this also pertains to my previous comment), if you want to make this
        # code work for cases where there are multiple target cells (which I
        # assume was once the case), then you could keep this here. However,
        # there would need to be several other changes to the codebase to
        # reflect this generality (e.g. in get_rec_vecs). I would recommend just
        # keeping it specific and changing these lines for now.
    )
    offTargetBound = (
        np.sum(
            cytBindingModel_vec(
                recCount=offTargRecs.to_numpy(),
                recXaffs=affs,
                dose=dose,
                valencies=valencies,
            )
        )
        / offTargRecs.shape[0]
    )

    return offTargetBound / targetBound


# Called in Figure1, Figure4, and Figure5
def optimizeDesign(  # rename to optimizeSelectivityAffs
    signal: str,
    targets: list,
    targCell: str,
    offTargCells: list,
    selectedDF: pd.DataFrame,
    dose: float,
    valencies: np.ndarray,
    prevOptAffs: list,  # rename to init_affinities
    cellCat="CellType2",
) -> tuple[float, float]:
    """
    A general purzse optimizer used to minimize selectivity output
        by varying affinity parameter.
    Args:
        signal: signaling receptor
        targets: list of targeting receptors
        targCell: target cell type
        offTargCells: list of off target cell types
        selectedDf: dataframe of receptor counts of all cells
        dose: ligand concentration/dose that is being modeled
        valencies: array of valencies of each ligand epitope
        prevOptAffs: initial receptor affinities to ultimately optimize for
            aximum target cell selectivity,
            affinities are K_a in L/mol
        epitope: additional epitope to be targeted
        cellCat: cell type categorization level, see cell types/subsets in CITE data

    Return:
        optSelectivity: optimized selectivity value
            Can also be modified to return optimized affinity parameter
    """

    X0 = prevOptAffs
    minAffs = (
        [7.0] * (len(targets) + 1)
    )  # minAffs and maxAffs chosen based on biologically realistic affinities for engineered ligands
    maxAffs = [9.0] * (len(targets) + 1)

    optBnds = Bounds(np.full_like(X0, minAffs), np.full_like(X0, maxAffs))

    dfTargCell = selectedDF.loc[
        selectedDF[cellCat] == targCell
    ]  # Originally performed using get_rec_vecs
    targRecs = dfTargCell[[signal] + targets + [cellCat]]
    dfOffTargCell = selectedDF.loc[selectedDF[cellCat].isin(offTargCells)]
    offTargRecs = dfOffTargCell[[signal] + targets + [cellCat]]

    optimized = minimize(
        minSelecFunc,
        X0,
        bounds=optBnds,
        args=(
            targRecs.drop([cellCat], axis=1),
            offTargRecs.drop([cellCat], axis=1),
            dose,
            valencies,
        ),
        jac="3-point",
    )
    optSelectivity = optimized.fun
    optAffs = optimized.x

    return optSelectivity, optAffs


# Called in all figures
def convFactCalc(CITE_DF: pd.DataFrame) -> pd.DataFrame:
    """
    Returns conversion factors by marker for converting CITEseq signal into abundance
    Args:
        CITE_DF: dataframe of unprocessed CITE-seq receptor values for each
            receptor (column) for each single cell (row)
    Return:
        weightDF: factor to convert unprocessed CITE-seq receptor values to
            numeric receptor counts
    """

    cellTypes = [
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
        for cell in cellTypes:
            cellTDF = CITE_DF.loc[CITE_DF["CellType2"] == cell][marker]
            dftemp = pd.DataFrame(
                {
                    "Marker": [marker],
                    "Cell Type": cell,
                    "Amount": cellTDF.mean(),
                    "Number": cellTDF.size,
                }
            )

            markerDF = (
                dftemp
                if isinstance(markerDF, pd.DataFrame)
                else pd.concat([markerDF, dftemp])
            )

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
    markerDF = markerDF.replace({"Marker": markDict, "Cell Type": cellDict})
    markerDFw = pd.DataFrame()

    # Armaan: You might need to step through the code to confirm this, but I
    # believe that this loop can be simplified to a loop over the rows in
    # markerDF, because there shouldn't be more than one row with the same cell
    # type and marker, this wouldn't even make sense (correct me if I'm wrong
    # though). This would allow you to avoid the use of unique(), the boolean &
    # indexing, and the use of wAvg.
    # Wait, actually, if my above statement is correct, then you can just delete
    # this whole double for loop and calculate the average in the loop above.
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

    # Armaan: If my comment above is correct, you can also move this logic into
    # the initial loop over markerDF and delete this loop.
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

        weightDF = dftemp if weightDF is None else pd.concat([weightDF, dftemp])

    return weightDF


# Called in Figure1, Figure3, Figure4, and Figure5
def getSampleAbundances(  # rename to calcReceptorAbundances
    epitopes: list, cellList: list, numCells=1000, cellCat="CellType2"
) -> pd.DataFrame:
    """
    Given list of epitopes and cell types, returns a dataframe
        containing receptor abundance data on a single-cell level.
    Args:
        epitopes: list of epitopes for which you want abundance values
        cellList: list of cell types for which you want epitope abundance
        numCells: number of cells to sample from for abundance calculations,
            default to sampling from 1000 cells
        cellCat: cell type categorization level, see cell types/subsets in CITE data
    Return:
        sampleDF: dataframe containing single cell abundances of
        receptors (column) for each individual cell (row),
        with final column being cell type from the cell type categorization level set by cellCat
    """

    CITE_DF = (
        importCITE()
    )  # Import CITE data and drop unnecessary epitopes and cell types
    CITE_DF_new = CITE_DF[epitopes + [cellCat]]
    CITE_DF_new = CITE_DF_new.loc[CITE_DF_new[cellCat].isin(cellList)]

    meanConv = (
        convFactCalc(CITE_DF).Weight.mean()
    )  # Get conv factors, average them to use on epitopes with unlisted conv facts
    # convFactDict values calculated by convFactCalc
    convFactDict = {
        "CD25": 77.136987,
        "CD122": 332.680090,
        "CD127": 594.379215,
    }  # move to convFactCalc?

    sampleDF = CITE_DF_new.sample(numCells, random_state=42)  # Sample df generated

    for epitope in epitopes:
        sampleDF[epitope] = sampleDF[epitope].multiply(
            convFactDict.get(epitope, meanConv)
        )

    return sampleDF


# Armaan: Can you refactor this and minOffTargSelec to use the same logic for
# inferring the number of bound signaling receptors? These two functions share a
# lot of the same functionality. One reason to avoid this is if this function is
# a lot slower, and you don't want to call it during optimization, but it
# doesn't seem obvious that it would be.
# Called in Figure1 and Figure3
def get_cell_bindings(
    df: np.ndarray,
    signal: str,
    targets: list,
    recXaffs: np.ndarray,
    dose: float,
    valencies: np.ndarray,
    cellCat="CellType2",
) -> pd.DataFrame:
    """
    Returns amount of receptor bound on average per cell for each cell type
    Args:
        df: dataframe of receptor counts of all cells
        signal: signaling receptor
        targets: list of targeting receptors
        recXaffs: receptor affinities
        dose: ligand concentration/dose that is being modeled
        valencies: array of valencies of each ligand epitope
        callCat: cell type categorization level, see cell types/subsets in CITE data
    Return:
        df_return: dataframe of average amount of receptor bound per cell (column) for
            each cell type (row)
    """

    targRecs = pd.DataFrame()
    df_return = pd.DataFrame()

    targRecs = df[[signal] + targets]
    affs = get_affs(recXaffs)

    cytBindingModel_vec = np.vectorize(  # Vectorization function for cytBindingModel
        cytBindingModel, excluded=["recXaffs", "valencies"], signature="(n),()->()"
    )

    targBound = cytBindingModel_vec(
        recCount=targRecs.to_numpy(), recXaffs=affs, dose=dose, valencies=valencies
    )

    df_return = df[[signal] + [cellCat]]
    df_return.insert(0, "Receptor Bound", targBound, True)
    df_return = df_return.groupby([cellCat]).mean(0)

    return df_return
