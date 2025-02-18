"""
Functions used in binding and selectivity analysis
"""

import numpy as np
from scipy.optimize import Bounds, minimize

from .binding_model_funcs import cyt_binding_model


# Called in minOffTargSelec and get_cell_bindings
# Sam: why not just input the affinities in the correct format...
def restructure_affs(affs: np.ndarray) -> np.ndarray:
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
def min_off_targ_selec(
    monomerAffs: np.ndarray,
    targRecs: np.ndarray,
    offTargRecs: np.ndarray,
    dose: float,
    valencies: np.ndarray,
) -> float:
    """
    Serves as the function which will have its return value
        minimized to get optimal selectivity.
        Used in conjunction with optimize_affs.
        The output (selectivity) is calculated based on the amounts of
        bound receptors of only the first column/receptor of
        the receptor abundance arrays.
    Args:
        monomerAffs: monomer ligand-receptor affinities
            Modulated in optimization
        targRecs: receptor counts of target cell type
        offTargRecs: receptors count of off-target cell types
        dose: ligand concentration/dose that is being modeled
        valencies: array of valencies of each ligand epitope
    Return:
        selectivity: value to be minimized.
            Defined as ratio of off target to on target binding.
            This is the selectivity for the off target cells, so is
            minimized to maximize selectivity for the target cell type.
    """

    assert targRecs.shape[1] == offTargRecs.shape[1]

    # Reformat input affinities
    modelAffs = restructure_affs(monomerAffs)

    # Use the binding model to calculate bound receptors
    #   for target and off-target cell types
    targRbound = cyt_binding_model(
        dose=dose, recCounts=targRecs, valencies=valencies, monomerAffs=modelAffs
    )
    offTargRbound = cyt_binding_model(
        dose=dose, recCounts=offTargRecs, valencies=valencies, monomerAffs=modelAffs
    )

    # Calculate total bound receptors for target and off-target
    #   cell types, normalized by number of cells
    targetBound = np.sum(targRbound[:, 0]) / targRbound.shape[0]
    offTargetBound = np.sum(offTargRbound[:, 0]) / offTargRbound.shape[0]

    # Return selectivity ratio
    return offTargetBound / targetBound


# Called in Figure1, Figure4, and Figure5
def optimize_affs(
    targRecs: np.ndarray,
    offTargRecs: np.ndarray,
    dose: float,
    valencies: np.ndarray,
    bounds: tuple[float, float] = (7.0, 9.0),
) -> tuple[float, list]:
    """
    An optimizer that maximizes the selectivity for a target cell type
        by varying the affinities of each receptor-ligand pair.
    Args:
        targRecs: receptor counts of each receptor (columns) on
            different cells (rows) of a target cell type. The
            first column must be the signal receptor which is used
            to calculate selectivity in min_off_targ_selec.
        offTargRecs: receptor counts of each receptor on
            different cells of off-target cell types. The
            columns must match the columns of targRecs.
        dose: ligand concentration/dose that is being modeled
        valencies: array of valencies of each ligand epitope
            Only set up for single complex modeling. Valencies
            must be a nested array, such as [[1, 1]] for a bivalent
            complex with two different ligands.
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
    #    for fig1, target 1, final affinities are 1e7 and ~1e9 (9.997e8) (bounds 7-9)
    minAffs = [bounds[0]] * (targRecs.shape[1])
    maxAffs = [bounds[1]] * (targRecs.shape[1])

    # Start at midpoint between min and max bounds
    # Sam: Correct this if sizes of initial affinities and valencies
    #   are not always the same
    initAffs = np.full_like(valencies[0], minAffs[0] + (maxAffs[0] - minAffs[0]) / 2)
    optBnds = Bounds(np.full_like(initAffs, minAffs), np.full_like(initAffs, maxAffs))

    targRecs[targRecs == 0] = 1e-9
    offTargRecs[offTargRecs == 0] = 1e-9

    # Run optimization to minimize off-target selectivity by changing affinities
    optimizer = minimize(
        fun=min_off_targ_selec,
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


# Called in Figure1 and Figure3
def get_cell_bindings(
    recCounts: np.ndarray, monomerAffs: np.ndarray, dose: float, valencies: np.ndarray
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
    modelAffs = restructure_affs(monomerAffs)

    # Use the binding model to calculate bound receptors for each cell
    Rbound = cyt_binding_model(
        dose=dose, recCounts=recCounts, valencies=valencies, monomerAffs=modelAffs
    )

    return Rbound
