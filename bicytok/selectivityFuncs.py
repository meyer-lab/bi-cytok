"""
Functions used in binding and selectivity analysis
"""

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, minimize

from bicytok.imports import importCITE, importReceptors
from bicytok.MBmodel import cytBindingModel


# Called in minOffTargSelec and get_cell_bindings
def restructureAffs(affs: np.ndarray) -> np.ndarray:
    """
    Structures array of receptor affinities to be compatible with the binding model
    Args:
        affs: receptor affinities
    Return:
        restructuredAffs: restructured receptor affinities
    """

    exponentialAffs = pd.DataFrame()
    # Convert affinities to 10th order values
    # Sam: why not just input the affinities in the correct format...
    for aff in affs:
        exponentialAffs = np.append(exponentialAffs, np.power(10, aff))

    # Sam: what is the meaning of the off-diagonal values?
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
    Serves as the function which will have its return value minimized to get optimal selectivity
        Used in conjunction with optimizeSelectivityAffs()
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

    # Reformat input affinities to 10^aff and diagonalize
    modelAffs = restructureAffs(monomerAffs)

    # Use the binding model to calculate bound receptors for target and off-target cell types
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

    # Calculate total bound receptors for target and off-target cell types, normalized by number of cells
    targetBound = np.sum(targRbound[:, 0]) / targRbound.shape[0]
    offTargetBound = np.sum(offTargRbound[:, 0]) / offTargRbound.shape[0]

    # Return selectivity ratio
    return offTargetBound / targetBound


# Called in Figure1, Figure4, and Figure5
def optimizeSelectivityAffs(
    initialAffs: list,
    targRecs: np.ndarray,
    offTargRecs: np.ndarray,
    dose: float,
    valencies: np.ndarray
) -> tuple[float, list]:
    """
    A general optimizer used to minimize selectivity output
        by varying affinity parameter.
    Args:
        initialAffs: initial receptor affinities to optimize for
            maximum target cell selectivity.
            affinities are K_a in L/mol
        targRecs: all receptor counts of target cell type
        offTargRecs: all receptor counts of off-target cell types
        dose: ligand concentration/dose that is being modeled
        valencies: array of valencies of each ligand epitope
    Return:
        optSelec: optimized selectivity value
        optAffs: optimized affinity values
    """

    # Relabel initial affinities to a variable that will be altered during optimization
    X0 = initialAffs

    # Set bounds for optimization
    # minAffs and maxAffs chosen based on biologically realistic affinities for engineered ligands
    # Sam: affinities are maxing and bottoming out before optimization is complete...
    #       for fig1, target 1, final affinities are 1e7 and ~1e9 (9.997e8)
    minAffs = [7.0] * (targRecs.shape[1])
    maxAffs = [9.0] * (targRecs.shape[1])
    optBnds = Bounds(np.full_like(X0, minAffs), np.full_like(X0, maxAffs))

    # Run optimization to minimize off-target selectivity by changing affinities
    optimizer = minimize(
        minOffTargSelec,
        X0,
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


# Called in sampleReceptorAbundances
# Sam: reimplement this function when we have a clearer idea 
#   of how to calculate conversion factors
def calcCITEConvFacts(CITE_DF: pd.DataFrame) -> pd.DataFrame:
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
    markDict = {
        "CD25": "IL2Ra", 
        "CD122": "IL2Rb", 
        "CD127": "IL7Ra", 
        "CD132": "gc"
    }
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
    
    return None


# Called in Figure1, Figure3, Figure4, and Figure5
def sampleReceptorAbundances(
    epitopes: list, 
    cellList: list, 
    numCells=1000, 
    cellCat="CellType2"
) -> pd.DataFrame:
    """
    Given list of epitopes and cell types, returns a dataframe
        containing receptor abundance data on a single-cell level.
    Args:
        epitopes: list of epitopes for which you want abundance values
        cellList: list of cell types for which you want epitope abundance
        numCells: number of cells to sample from for abundance calculations
        cellCat: cell type categorization level, see cell types/subsets in CITE data
    Return:
        sampleDF: dataframe containing single cell abundances of
            receptors (column) for each individual cell (row),
            with final column being cell type from the cell type categorization level set by cellCat
    """

    # Import CITE data and drop unused epitopes and cell types
    CITE_DF = (importCITE()) 
    CITE_DF_new = CITE_DF[epitopes + [cellCat]]
    CITE_DF_new = CITE_DF_new.loc[CITE_DF_new[cellCat].isin(cellList)]
    CITE_DF_new = CITE_DF_new.rename(columns={cellCat: "Cell Type"})

    # convFactDict values calculated by calcCITEConvFacts
    # Sam: calculation of conversion factors was unclear, should be revised
    convFactDict = {
        "CD25": 77.136987,
        "CD122": 332.680090,
        "CD127": 594.379215,
    } 
    
    # Sample a subset of cells
    sampleDF = CITE_DF_new.sample(numCells, random_state=42)  # Sample df generated

    # Multiply the receptor counts of epitope by the conversion factor for that epitope
    for epitope in epitopes:
        sampleDF[epitope] = sampleDF[epitope].multiply(
            convFactDict.get(epitope, value=300) # If epitope not in dict, multiply by 300
        )

    return sampleDF


# Called in Figure1 and Figure3
def get_cell_bindings(
    recCounts: np.ndarray,
    monomerAffs: np.ndarray,
    dose: float,
    valencies: np.ndarray
) -> pd.DataFrame:
    """
    Returns amount of receptor bound to each cell
    Args:
        recCounts: counts of each receptor on all single cells
        monomerAffs: monomer ligand-receptor affinities
        dose: ligand concentration/dose that is being modeled
        valencies: array of valencies of each ligand epitope
    Return:
        df_return: dataframe of average amount of receptor bound per cell (column) for
            each cell type (row)
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