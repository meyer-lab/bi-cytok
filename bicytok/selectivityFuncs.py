"""
Functions used in binding and selectivity analysis
"""

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, minimize

from .MBmodel import cytBindingModel


# Called in minOffTargSelec and get_cell_bindings
# Sam: why not just input the affinities in the correct format...
def restructureAffs(affs: np.ndarray) -> np.ndarray:
    """
    Structures array of receptor affinities to be compatible with the binding model
    Args:
        affs: receptor affinities in ? units
    Return:
        restructuredAffs: restructured receptor affinities in L/mol (1/M)
    """

    assert len(affs.shape) == 1
    assert affs.size > 0

    # Convert affinities to 10th order values
    exponentialAffs = np.power(10, affs)

    # Set off-diagonals to values that won't affect optimization
    restructuredAffs = np.full((affs.size, affs.size), 1e2)
    np.fill_diagonal(restructuredAffs, exponentialAffs)

    return restructuredAffs


# Called in optimizeDesign
def minOffTargSelec(
    monomerAffs: np.ndarray,
    targRecs: np.ndarray,
    offTargRecs: np.ndarray,
    dose: float,
    valencies: np.ndarray,
) -> float:
    """
    Serves as the function which will have its return value 
        minimized to get optimal selectivity.
        Used in conjunction with optimizeSelectivityAffs().
    Args:
        monomerAffs: monomer ligand-receptor affinities 
            Modulated in optimization
        targRecs: dataframe of receptors counts of target cell type
        offTargRecs: dataframe of receptors counts of off-target cell types
        dose: ligand concentration/dose that is being modeled
        valencies: array of valencies of each ligand epitope
    Return:
        selectivity: value will be minimized, 
            defined as ratio of off target to on target signaling,
            this is selectivity for the off target cells and is minimized to
            maximize selectivity for the target cell
    """

    assert targRecs.shape[1] == offTargRecs.shape[1]

    # Reformat input affinities to 10^aff and diagonalize
    # Sam: shouldn't this be done before the optimization?
    modelAffs = restructureAffs(monomerAffs)

    # Use the binding model to calculate bound receptors 
    #   for target and off-target cell types
    targRbound = cytBindingModel(
        dose=dose,
        recCounts=targRecs,
        valencies=valencies,
        monomerAffs=modelAffs
    )
    offTargRbound = cytBindingModel(
        dose=dose,
        recCounts=offTargRecs,
        valencies=valencies,
        monomerAffs=modelAffs
    )

    # Calculate total bound receptors for target and off-target 
    #   cell types, normalized by number of cells
    targetBound = np.sum(targRbound[:, 0]) / targRbound.shape[0]
    offTargetBound = np.sum(offTargRbound[:, 0]) / offTargRbound.shape[0]

    # Return selectivity ratio
    return offTargetBound / targetBound


# Called in Figure1, Figure4, and Figure5
def optimizeSelectivityAffs(
    targRecs: np.ndarray,
    offTargRecs: np.ndarray,
    dose: float,
    valencies: np.ndarray,
    bounds: tuple[float, float] = (7.0, 9.0)
) -> tuple[float, list]:
    """
    An optimizer used to minimize selectivity output
        by varying the affinity parameters.
        Selectivity is defined as the ratio of binding of
        off-target cells to target cells.
    Args:
        targRecs: receptor counts of each receptor (columns) on 
            different cells (rows) of target cell type
        offTargRecs: receptor counts of each receptor on
            different cells of off-target cell types
        dose: ligand concentration/dose that is being modeled
        valencies: array of valencies of each ligand epitope
        bounds: minimum and maximum optimization bounds for affinity values
    Return:
        optSelec: optimized selectivity value
        optAffs: optimized affinity values that yield the optimized selectivity
    """    

    assert targRecs.size > 0
    assert offTargRecs.size > 0

    # Choose initial affinities and set bounds for optimization
    # minAffs and maxAffs chosen based on biologically realistic affinities 
    #   for engineered ligands
    # Sam: affinities are maxing and bottoming out before optimization is complete...
    #    for fig1, target 1, final affinities are 1e7 and ~1e9 
    #    (9.997e8) (with bounds 7 and 9)
    # Sam: need to test this for more than two epitopes
    minAffs = [bounds[0]] * (targRecs.shape[1])
    maxAffs = [bounds[1]] * (targRecs.shape[1])

    # Start at midpoint between min and max bounds
    # Sam: Correct this if sizes of initial affinities and valencies are not always the same
    initAffs = np.full_like(
        valencies[0], 
        minAffs[0] + (maxAffs[0] - minAffs[0]) / 2
    )
    optBnds = Bounds(
        np.full_like(initAffs, minAffs), 
        np.full_like(initAffs, maxAffs)
    )

    # Run optimization to minimize off-target selectivity by changing affinities
    optimizer = minimize(
        fun=minOffTargSelec,
        x0=initAffs,
        bounds=optBnds,
        args=(
            targRecs,
            offTargRecs,
            dose,
            valencies,
        ),
        jac="3-point",
    )
    optSelect = optimizer.fun
    optAffs = optimizer.x

    return optSelect, optAffs


# Sam: reimplement this function when we have a clearer idea 
#   of how to calculate conversion factors
def calcCITEConvFacts() -> tuple[dict, float]:
    """
    Returns conversion factors by marker for converting CITEseq signal into abundance
    """

    # cellTypes = [
    #     "CD4 TCM",
    #     "CD8 Naive",
    #     "NK",
    #     "CD8 TEM",
    #     "CD4 Naive",
    #     "CD4 CTL",
    #     "CD8 TCM",
    #     "Treg",
    #     "CD4 TEM",
    # ]
    # markers = ["CD122", "CD127", "CD25"]
    # markDict = {
    #     "CD25": "IL2Ra", 
    #     "CD122": "IL2Rb", 
    #     "CD127": "IL7Ra", 
    #     "CD132": "gc"
    # }
    # cellDict = {
    #     "CD4 Naive": "Thelper",
    #     "CD4 CTL": "Thelper",
    #     "CD4 TCM": "Thelper",
    #     "CD4 TEM": "Thelper",
    #     "NK": "NK",
    #     "CD8 Naive": "CD8",
    #     "CD8 TCM": "CD8",
    #     "CD8 TEM": "CD8",
    #     "Treg": "Treg",
    # }

    # Sam: calculation of these conversion factors was unclear, should be revised
    origConvFactDict = {
        "CD25": 77.136987,
        "CD122": 332.680090,
        "CD127": 594.379215,
    } 
    convFactDict = origConvFactDict.copy()
    defaultConvFact = np.mean(list(origConvFactDict.values()))

    return convFactDict, defaultConvFact


# Called in Figure1, Figure3, Figure4, and Figure5
def sampleReceptorAbundances(
    CITE_DF: pd.DataFrame,
    numCells=1000, 
) -> pd.DataFrame:
    """
    Samples a subset of cells and converts unprocessed CITE-seq receptor values 
        into abundance values.
    Args:
        CITE_DF: dataframe of unprocessed CITE-seq receptor counts 
            of different receptors/epitopes (columns) on single cells (row).
            Epitopes are filtered outside of this function.
            The final column should be the cell types of each cell.
        numCells: number of cells to sample
    Return:
        sampleDF: dataframe containing single cell abundances of
            receptors (column) for each individual cell (row).
            The final column is the cell type of each cell.
    """

    assert numCells <= CITE_DF.shape[0]
    assert "Cell Type" in CITE_DF.columns
    
    # Sample a subset of cells
    sampleDF = CITE_DF.sample(numCells, random_state=42)

    # Calculate conversion factors for each epitope
    convFactDict, defaultConvFact = calcCITEConvFacts()

    # Multiply the receptor counts of epitope by the conversion factor for that epitope
    epitopes = CITE_DF.columns[CITE_DF.columns != "Cell Type"]
    convFacts = [convFactDict.get(epitope, defaultConvFact) 
                 for epitope 
                 in epitopes]

    sampleDF[epitopes] = sampleDF[epitopes] * convFacts
    
    return sampleDF


# Called in Figure1 and Figure3
def get_cell_bindings(
    recCounts: np.ndarray,
    monomerAffs: np.ndarray,
    dose: float,
    valencies: np.ndarray
) -> np.ndarray:
    """
    Returns amount of receptor bound to each cell
    Args:
        recCounts: single cell abundances of receptors
        monomerAffs: monomer ligand-receptor affinities
        dose: ligand concentration/dose that is being modeled
        valencies: array of valencies of each ligand epitope
    Return:
        Rbound: number of bound receptors for each cell
    """

    # Reformat input affinities to 10^aff and diagonalize
    modelAffs = restructureAffs(monomerAffs)

    # Use the binding model to calculate bound receptors for each cell
    Rbound = cytBindingModel(
        dose=dose, 
        recCounts=recCounts,
        valencies=valencies, 
        monomerAffs=modelAffs
    )

    return Rbound