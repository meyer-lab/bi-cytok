"""
Functions used in binding and selectivity analysis
"""

import numpy as np
from scipy.optimize import Bounds, minimize

from .binding_model_funcs import cyt_binding_model


# Called in minOffTargSelec and get_cell_bindings
def restructure_affs(affs: np.ndarray) -> np.ndarray:
    """
    Structures array of receptor affinities to be compatible with the binding model
    Args:
        affs: receptor affinities in log10(M)
    Return:
        restructuredAffs: restructured receptor affinities in L/mol (1/M)
    """

    assert len(affs.shape) == 1
    assert affs.size > 0

    # Convert affinities to 10th order values
    exponentialAffs = np.power(10, affs)

    # Set off-diagonals to values that won't affect optimization
    restructuredAffs = np.full((affs.size, affs.size), 0.0)
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

    # Use binding model to calculate bound receptors for target and off-target cells
    targRbound = cyt_binding_model(
        dose=dose, recCounts=targRecs, valencies=valencies, monomerAffs=modelAffs
    )
    offTargRbound = cyt_binding_model(
        dose=dose, recCounts=offTargRecs, valencies=valencies, monomerAffs=modelAffs
    )

    # Calculate mean bound receptors for target and off-target cell types
    targetBound = np.sum(targRbound[:, 0]) / targRbound.shape[0]
    offTargetBound = np.sum(offTargRbound[:, 0]) / offTargRbound.shape[0]

    # Return selectivity ratio
    return offTargetBound / targetBound


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
    minAffs = [bounds[0]] * (targRecs.shape[1])
    maxAffs = [bounds[1]] * (targRecs.shape[1])

    # Start at midpoint between min and max bounds
    initAffs = np.full_like(valencies[0], minAffs[0] + (maxAffs[0] - minAffs[0]) / 2)
    optBnds = Bounds(np.full_like(initAffs, minAffs), np.full_like(initAffs, maxAffs))

    # Normalize all target receptor counts to the mean signal receptor count
    if targRecs.shape[1] > 1:
        fullRecs = np.concatenate((targRecs, offTargRecs), axis=0)
        signalMean = np.mean(fullRecs[:, 0])
        for i in range(1, targRecs.shape[1]):
            receptorMean = np.mean(fullRecs[:, i])
            targRecs[:, i] = targRecs[:, i] * (signalMean / receptorMean)
            offTargRecs[:, i] = offTargRecs[:, i] * (signalMean / receptorMean)

    # Set zero counts to a small value to avoid division by zero
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

    recCounts[recCounts == 0] = 1e-9

    # Use the binding model to calculate bound receptors for each cell
    Rbound = cyt_binding_model(
        dose=dose, recCounts=recCounts, valencies=valencies, monomerAffs=modelAffs
    )

    return Rbound
