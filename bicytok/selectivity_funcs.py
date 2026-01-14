"""
Functions used in binding and selectivity analysis
"""

import os

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
from jaxtyping import Array, Float64, Scalar

from .binding_model_funcs import cyt_binding_model

jax.config.update("jax_enable_x64", True)
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".05"


def restructure_affs(
    affs: Float64[Array, "receptors"],
) -> Float64[Array, "receptors receptors"]:
    """
    Structures array of receptor affinities to be compatible with the binding model.

    Arguments:
        affs: receptor affinities in log10(M)

    Outputs:
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


def min_off_targ_selec(
    params: Float64[Array, "receptors_plus_one"],
    targRecs: Float64[Array, "cells receptors"],
    offTargRecs: Float64[Array, "cells receptors"],
    dose: Scalar,
    valencies: Float64[Array, "receptors"],
) -> Scalar:
    """
    The objective function to optimize selectivity by varying affinities. The output
        (selectivity) is calculated based on the amounts of bound receptors of only the
        first column/receptor (i.e., the signal receptor).

    Arguments:
        params: combined array of [monomer affinities, log10(Kx_star)]
        targRecs: receptor counts of target cell type
        offTargRecs: receptors count of off-target cell types
        dose: ligand concentration/dose
        valencies: array of valencies of each distinct ligand in the ligand complex

    Outputs:
        selectivity: value to be minimized. Defined as ratio of off target to on target
            binding. By minimizing the selectivity for off-target cells, we maximize
            the selectivity for target cells.
    """

    assert targRecs.shape[1] == offTargRecs.shape[1]

    monomerAffs = params[:-1]

    # Reformat input affinities
    modelAffs = restructure_affs(monomerAffs)
    Kx_star = jnp.power(10, params[-1])

    # Use the binding model to calculate bound receptors
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

    # Calculate total bound receptors for target and off-target cell types, normalized
    #   by number of cells
    targetBound = jnp.sum(targRbound[:, 0]) / targRbound.shape[0]
    offTargetBound = jnp.sum(offTargRbound[:, 0]) / offTargRbound.shape[0]

    # Return selectivity ratio
    return (targetBound + offTargetBound) / targetBound


# Affinity optimization constants
REC_COUNT_EPS = 1e-6


def _optimize_affs_jax(
    targRecs_jax: Float64[Array, "cells receptors"],
    offTargRecs_jax: Float64[Array, "cells receptors"],
    dose: Scalar,
    valencies_jax: Float64[Array, "receptors"],
    affinity_bounds: tuple[float, float],
    Kx_star_bounds: tuple[float, float],
    max_iter: int,
    tol: float,
    init_params: Float64[Array, "receptors_plus_one"],
) -> tuple[Scalar, Float64[Array, "receptors"], Scalar]:
    """
    JAX-optimized core optimization function using L-BFGS-B.
    """

    minAffs = jnp.full(targRecs_jax.shape[1], affinity_bounds[0])
    maxAffs = jnp.full(targRecs_jax.shape[1], affinity_bounds[1])

    # Set optimization bounds
    bounds = (
        jnp.concatenate(
            [minAffs, jnp.array([Kx_star_bounds[0]])]
        ),  # lower bounds
        jnp.concatenate(
            [maxAffs, jnp.array([Kx_star_bounds[1]])]
        ),  # upper bounds
    )

    # Replace zero receptor counts with small epsilon to avoid instability
    targRecs_clean = jnp.where(targRecs_jax == 0, REC_COUNT_EPS, targRecs_jax)
    offTargRecs_clean = jnp.where(offTargRecs_jax == 0, REC_COUNT_EPS, offTargRecs_jax)

    # Set up L-BFGS-B solver from jaxopt
    solver = jaxopt.LBFGSB(fun=min_off_targ_selec, maxiter=max_iter, tol=tol)

    # Run optimization with bounds passed to run method
    result = solver.run(
        init_params=init_params,
        bounds=bounds,
        targRecs=targRecs_clean,
        offTargRecs=offTargRecs_clean,
        dose=dose,
        valencies=valencies_jax,
    )

    final_params = result.params
    final_loss = result.state.value

    return final_loss, final_params[:-1], jnp.power(10, final_params[-1])


def optimize_affs(
    targRecs: np.ndarray,
    offTargRecs: np.ndarray,
    dose: float,
    valencies: np.ndarray,
    init_vals: np.ndarray | str | int = 42,
    affinity_bounds: tuple[float, float] = (6.0, 12.0),
    Kx_star_bounds: tuple[float, float] = (-15, -9),
    max_iter: int = 1000,
    tol: float = 1e-5,
) -> tuple[float, list, float]:
    """
    NumPy-compatible wrapper for optimize_affs that handles conversions to/from JAX.
    Minimizes the off-target to on-target selectivity ratio by optimizing
    receptor affinities and Kx_star using L-BFGS.

    Arguments:
        targRecs: receptor counts of target cell type
        offTargRecs: receptors count of off-target cell types
        dose: ligand concentration/dose
        valencies: array of valencies of each distinct ligand in the ligand complex
        init_vals: seed for random initial parameters or array of initial parameters
        affinity_bounds: minimum and maximum optimization bounds for affinity values
        Kx_star_bounds: minimum and maximum optimization bounds for Kx_star
        max_iter: maximum number of iterations
        tol: parameter tolerance for convergence

    Outputs:
        optSelec: optimized selectivity value
        optAffs: optimized affinity values
        optKx_star: optimized Kx_star value
    """

    assert targRecs.size > 0
    assert offTargRecs.size > 0
    assert targRecs.shape[1] == offTargRecs.shape[1]

    # Set up initial parameters
    if isinstance(init_vals, int):
        rng = np.random.default_rng(seed=init_vals)
        init_affs = rng.uniform(
            low=affinity_bounds[0],
            high=affinity_bounds[1],
            size=targRecs.shape[1],
        )
        init_kx_star = rng.uniform(
            low=Kx_star_bounds[0], high=Kx_star_bounds[1]
        )
        init_params = np.concatenate((init_affs, np.array([init_kx_star])))
    elif isinstance(init_vals, str) and init_vals == "search":
        init_params, init_selectivity = search_initialization(
            targ_counts=targRecs,
            off_targ_counts=offTargRecs,
            dose=dose,
            valencies=valencies,
            affinity_bounds=affinity_bounds,
            Kx_star_bounds=Kx_star_bounds,
        )
    else:
        init_params = init_vals

    # Convert inputs to JAX arrays
    targRecs_jax = jnp.array(targRecs, dtype=jnp.float64)
    dose_jax = jnp.array(dose, dtype=jnp.float64)
    offTargRecs_jax = jnp.array(offTargRecs, dtype=jnp.float64)
    valencies_jax = jnp.array(valencies, dtype=jnp.float64)
    init_params_jax = jnp.array(init_params, dtype=jnp.float64)

    _optimize_affs_jax_jit = jax.jit(_optimize_affs_jax)

    # Call the JAX-optimized function
    final_loss, final_affs, final_kx_star = _optimize_affs_jax_jit(
        targRecs_jax,
        offTargRecs_jax,
        dose_jax,
        valencies_jax,
        affinity_bounds,
        Kx_star_bounds,
        max_iter,
        tol,
        init_params_jax,
    )

    # Convert outputs back to Python types
    if isinstance(init_vals, str) and init_vals == "search":
        return (
            float(final_loss),
            [float(x) for x in final_affs],
            float(final_kx_star),
            init_params,
            init_selectivity,
        )    
    else:
        return (
            float(final_loss),
            [float(x) for x in final_affs],
            float(final_kx_star),
        )


cyt_binding_model_jit = jax.jit(cyt_binding_model)


def get_cell_bindings(
    recCounts: np.ndarray,
    monomerAffs: np.ndarray,
    dose: float,
    valencies: np.ndarray,
    Kx_star: float = -12,
) -> np.ndarray:
    """
    Predicts the amount of bound receptors across cells based on set affinities.

    Arguments:
        recCounts: single cell abundances of receptors
        monomerAffs: monomer ligand-receptor affinities
        dose: ligand concentration/dose that is being modeled
        valencies: array of valencies of each distinct ligand in the ligand complex
        Kx_star: cross-linking constant for the binding model

    Outputs:
        Rbound: number of bound receptors for each cell
    """

    monomerAffs = jnp.array(monomerAffs, dtype=jnp.float64)
    recCounts = jnp.array(recCounts, dtype=jnp.float64)
    valencies = jnp.array(valencies, dtype=jnp.float64)

    # Reformat input affinities to 10^aff and diagonalize
    modelAffs = restructure_affs(monomerAffs)
    Kx_star_mod = jnp.power(10, Kx_star)

    # Use the binding model to calculate bound receptors for each cell
    Rbound = cyt_binding_model_jit(
        dose=dose,
        recCounts=recCounts,
        valencies=valencies,
        monomerAffs=modelAffs,
        Kx_star=Kx_star_mod,
    )

    return Rbound


def compute_selectivity(
    targ_counts: np.ndarray,
    off_targ_counts: np.ndarray,
    affinities: np.ndarray,
    dose: float,
    valencies: np.ndarray,
    Kx_star: float = -12,
    method: str = "mean",
) -> float:
    """
    Computes the selectivity based on given affinities and Kx_star.

    Arguments:
        targ_counts: receptor counts of target cell type
        off_targ_counts: receptors count of off-target cell types
        affinities: array of receptor affinities in log10(M)
        dose: ligand concentration/dose
        valencies: array of valencies of each distinct ligand in the ligand complex
        Kx_star: cross-linking constant for the binding model

    Outputs:
        selectivity: ratio of off-target to on-target binding
    """

    # Get bound receptors for target and off-target cell types
    targ_bound = np.array(
        get_cell_bindings(
            recCounts=targ_counts,
            monomerAffs=affinities,
            dose=dose,
            valencies=valencies,
            Kx_star=Kx_star,
        )
    )
    off_targ_bound = np.array(
        get_cell_bindings(
            recCounts=off_targ_counts,
            monomerAffs=affinities,
            dose=dose,
            valencies=valencies,
            Kx_star=Kx_star,
        )
    )

    # Summarize bound receptors based on method
    if method == "mean":
        targetBound = np.sum(targ_bound[:, 0]) / targ_bound.shape[0]
        offTargetBound = np.sum(off_targ_bound[:, 0]) / off_targ_bound.shape[0]
    elif method == "geometric_mean":
        targetBound = np.exp(np.mean(np.log(targ_bound[:, 0] + 1e-12)))
        offTargetBound = np.exp(np.mean(np.log(off_targ_bound[:, 0] + 1e-12)))

    # Return selectivity ratio
    return (targetBound + offTargetBound) / targetBound


def search_initialization(
    targ_counts: np.ndarray,
    off_targ_counts: np.ndarray,
    dose: float,
    valencies: np.ndarray,
    grid_size: int = 5,
    affinity_bounds: tuple[float, float] = (6.0, 12.0),
    Kx_star_bounds: tuple[float, float] = (-15, -9),
) -> list:
    """

    """
    # Create grid of initial affinities and Kx_star values
    signal_aff_grid = np.linspace(
        affinity_bounds[0], affinity_bounds[1], grid_size
    )
    target_aff_grid = np.linspace(
        affinity_bounds[0], affinity_bounds[1], grid_size
    )
    Kx_star_grid = np.linspace(
        Kx_star_bounds[0], Kx_star_bounds[1], grid_size
    )

    optimal_loss = np.inf
    optimal_initialization = None

    for sig_aff in signal_aff_grid:
        for targ_aff in target_aff_grid:
            for Kx in Kx_star_grid:
                selectivity = compute_selectivity(
                    targ_counts=targ_counts,
                    off_targ_counts=off_targ_counts,
                    affinities=np.array([sig_aff, targ_aff, targ_aff]),
                    dose=dose,
                    valencies=valencies,
                    Kx_star=Kx,
                )
                if selectivity < optimal_loss:
                    optimal_loss = selectivity
                    optimal_initialization = np.array(
                        [sig_aff, targ_aff, targ_aff, Kx]
                    )
            
    return optimal_initialization, optimal_loss