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
    restructuredAffs = np.full((affs.size, affs.size), 1e-9)
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
        bound receptors of the first two columns/receptors (signal and target)
        of the receptor abundance arrays, averaged together.
    Args:
        monomerAffs: monomer ligand-receptor affinities
            Modulated in optimization
        targRecs: receptor counts of target cell type
        offTargRecs: receptors count of off-target cell types
        dose: ligand concentration/dose that is being modeled
        valencies: array of valencies of each ligand epitope
    Return:
        selectivity: value to be minimized.
            Defined as average of signal and target receptor selectivity ratios.
            This is the selectivity for the off target cells, so is
            minimized to maximize selectivity for the target cell type.
    """

    assert targRecs.shape[1] == offTargRecs.shape[1]
    assert targRecs.shape[1] >= 2  # Need at least signal and target receptors

    # Reformat input affinities
    modelAffs = restructure_affs(monomerAffs)

    # Use binding model to calculate bound receptors for target and off-target cells
    targRbound, _ = cyt_binding_model(
        dose=dose, recCounts=targRecs, valencies=valencies, monomerAffs=modelAffs
    )
    offTargRbound, _ = cyt_binding_model(
        dose=dose, recCounts=offTargRecs, valencies=valencies, monomerAffs=modelAffs
    )

    fullRbound = np.concatenate((targRbound, offTargRbound), axis=0)

    # Calculate mean bound receptors for signal receptor (column 0)
    targetBoundSignal = np.sum(targRbound[:, 0]) / targRbound.shape[0]
    fullBoundSignal = np.sum(offTargRbound[:, 0]) / offTargRbound.shape[0]

    # Calculate mean bound receptors for target receptor (column 1)
    targetBoundTarget = np.sum(targRbound[:, 1]) / targRbound.shape[0]
    fullBoundTarget = np.sum(offTargRbound[:, 1]) / offTargRbound.shape[0]

    # Calculate selectivity for each receptor
    signalSelectivity = fullBoundSignal / targetBoundSignal
    targetSelectivity = fullBoundTarget / targetBoundTarget

    # Return average selectivity ratio
    return (1*signalSelectivity + 1*targetSelectivity) / 2


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
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns amount of receptor bound to each cell and the binding model losses
    Args:
        recCounts: single cell abundances of receptors
        monomerAffs: monomer ligand-receptor affinities
        dose: ligand concentration/dose that is being modeled
        valencies: array of valencies of each ligand epitope
    Return:
        Rbound: number of bound receptors for each cell
        losses: binding model optimization losses for each cell
    """

    # Reformat input affinities to 10^aff and diagonalize
    modelAffs = restructure_affs(monomerAffs)

    recCounts[recCounts == 0] = 1e-9

    # Use the binding model to calculate bound receptors for each cell
    Rbound, losses = cyt_binding_model(
        dose=dose, recCounts=recCounts, valencies=valencies, monomerAffs=modelAffs
    )

    return Rbound, losses
