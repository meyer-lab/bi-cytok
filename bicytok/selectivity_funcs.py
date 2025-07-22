"""
Functions used in binding and selectivity analysis
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import Bounds, minimize

from .binding_model_funcs import cyt_binding_model


# Called in minOffTargSelec and get_cell_bindings
def restructure_affs(affs: jnp.ndarray) -> jnp.ndarray:
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
    exponentialAffs = jnp.power(10, affs)

    # Set off-diagonals to values that won't affect optimization
    restructuredAffs = jnp.full((affs.size, affs.size), 0.0)
    restructuredAffs = jnp.fill_diagonal(
        restructuredAffs, exponentialAffs, inplace=False
    )

    return restructuredAffs


# Called in optimizeDesign
def min_off_targ_selec(
    params: jnp.ndarray,
    targRecs: jnp.ndarray,
    offTargRecs: np.ndarray,
    dose: float,
    valencies: jnp.ndarray,
):
    """
    Serves as the function which will have its return value
        minimized to get optimal selectivity.
        Used in conjunction with optimize_affs.
        The output (selectivity) is calculated based on the amounts of
        bound receptors of only the first column/receptor of
        the receptor abundance arrays.
    Args:
        params: combined array of [monomer affinities, log10(Kx_star)]
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

    monomerAffs = params[:-1]
    Kx_star = jnp.power(10, params[-1])

    # Reformat input affinities
    modelAffs = restructure_affs(monomerAffs)

    # Use the binding model to calculate bound receptors
    #   for target and off-target cell types
    targRbound = cyt_binding_model(
        dose=dose,
        recCounts=targRecs,
        valencies=valencies,
        monomerAffs=modelAffs,
        Kx_star=Kx_star,
    )
    offTargRbound = cyt_binding_model(
        dose=dose,
        recCounts=offTargRecs,
        valencies=valencies,
        monomerAffs=modelAffs,
        Kx_star=Kx_star,
    )

    # Calculate total bound receptors for target and off-target
    #   cell types, normalized by number of cells
    targetBound = np.sum(targRbound[:, 0]) / targRbound.shape[0]
    offTargetBound = np.sum(offTargRbound[:, 0]) / offTargRbound.shape[0]

    # Return selectivity ratio
    return (targetBound + offTargetBound) / targetBound


# Define the function to minimize, which is the selectivity
min_off_targ_selec_jax = jax.jit(jax.value_and_grad(min_off_targ_selec))
min_off_targ_selec_jax_hess = jax.hessian(min_off_targ_selec)


def optimize_affs(
    targRecs: np.ndarray,
    offTargRecs: np.ndarray,
    dose: float,
    valencies: np.ndarray,
    affinity_bounds: tuple[float, float] = (7.0, 9.0),
    Kx_star_bounds: tuple[float, float] = (2.24e-13, 2.24e-11),
) -> tuple[float, list, float]:
    """
    An optimizer that maximizes the selectivity for a target cell type
        by varying the affinities of each receptor-ligand pair and the
        cross-linking constant Kx_star.
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
        affinity_bounds: minimum and maximum optimization bounds for affinity values
        Kx_star_bounds: minimum and maximum optimization bounds for Kx_star
    Return:
        optSelec: optimized selectivity value
        optAffs: optimized affinity values that yield the optimized selectivity
        optKx_star: optimized Kx_star value
    """

    assert targRecs.size > 0
    assert offTargRecs.size > 0

    minAffs = [affinity_bounds[0]] * (targRecs.shape[1])
    maxAffs = [affinity_bounds[1]] * (targRecs.shape[1])

    # Start at midpoint between min and max bounds
    initAffs = np.full_like(valencies[0], minAffs[0] + (maxAffs[0] - minAffs[0]) / 2)
    initKx_star = np.log10(
        Kx_star_bounds[0] + (Kx_star_bounds[1] - Kx_star_bounds[0]) / 2
    )
    initParams = np.concatenate((initAffs, [initKx_star]))

    # Set bounds for optimization
    minBounds = np.concatenate([minAffs, [np.log10(Kx_star_bounds[0])]])
    maxBounds = np.concatenate([maxAffs, [np.log10(Kx_star_bounds[1])]])
    optBnds = Bounds(minBounds, maxBounds)

    targRecs[targRecs == 0] = 1e-9
    offTargRecs[offTargRecs == 0] = 1e-9

    # Run optimization to minimize off-target selectivity by changing affinities and Kx_star
    optimizer = minimize(
        fun=min_off_targ_selec_jax,
        hess=min_off_targ_selec_jax_hess,
        method="trust-constr",
        x0=initParams,
        bounds=optBnds,
        args=(
            targRecs,
            offTargRecs,
            dose,
            valencies,
        ),
        jac=True,
        options={"disp": False, "xtol": 1e-12, "gtol": 1e-12},
    )
    optSelect = optimizer.fun
    optAffs = optimizer.x[:-1]
    optKx_star = np.power(10, optimizer.x[-1])
    convergence = optimizer.success

    if not convergence or optSelect < 0 or optSelect > 1:
        print(
            f"Optimization warning: {optimizer.message}, "
            f"Selectivity: {optSelect:.3f}, affinity values: {optAffs}, "
            f"cross-linking constant: {optKx_star:.2e}, Convergence: {convergence}"
        )

    return optSelect, optAffs, optKx_star


def get_cell_bindings(
    recCounts: np.ndarray,
    monomerAffs: np.ndarray,
    dose: float,
    valencies: np.ndarray,
    Kx_star: float = 2.24e-12,
) -> np.ndarray:
    """
    Returns amount of receptor bound to each cell
    Args:
        recCounts: single cell abundances of receptors
        monomerAffs: monomer ligand-receptor affinities
        dose: ligand concentration/dose that is being modeled
        valencies: array of valencies of each ligand epitope
        Kx_star: cross-linking constant for the binding model
    Return:
        Rbound: number of bound receptors for each cell
    """

    # Reformat input affinities to 10^aff and diagonalize
    modelAffs = restructure_affs(monomerAffs)

    # Use the binding model to calculate bound receptors for each cell
    Rbound = cyt_binding_model(
        dose=dose,
        recCounts=recCounts,
        valencies=valencies,
        monomerAffs=modelAffs,
        Kx_star=Kx_star,
    )

    return Rbound
