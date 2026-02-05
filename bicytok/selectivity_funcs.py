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
    reg_weight: float,
):
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
        reg_weight: regularization weight for variance in target binding

    Outputs:
        selectivity: value to be minimized. Defined as ratio of off target to on target
            binding. By minimizing the selectivity for off-target cells, we maximize
            the selectivity for target cells. Can include regularization term to
            penalize high variance in target binding.
    """

    assert targRecs.shape[1] == offTargRecs.shape[1]

    monomerAffs = params[:-1]
    Kx_star = jnp.power(10, params[-1])

    # Reformat input affinities
    modelAffs = restructure_affs(monomerAffs)

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
    n_targets = targRbound.shape[0]
    n_off_targets = offTargRbound.shape[0]
    targetBound = jnp.exp(jnp.sum(jnp.log(targRbound[:, 0]) / n_targets))
    offTargetBound = jnp.exp(jnp.sum(jnp.log(offTargRbound[:, 0]) / n_off_targets))

    # reg_metric = jnp.var(targRbound[:, 0]) + 1 / jnp.mean(targRbound[:, 0]) # Variance + inverse mean
    reg_metric = 1 / jnp.mean(targRbound[:, 0])  # Inverse mean
    # reg_metric = jnp.median(jnp.abs(targRbound[:, 0] - jnp.median(targRbound[:, 0]))) # MAD
    # reg_metric = jnp.subtract(*jnp.percentile(targRbound[:, 0], [75, 25])) # IQR

    # Return selectivity ratio
    return (targetBound + offTargetBound) / targetBound + reg_weight * reg_metric

# Affinity optimization constants
INIT_AFF_SEED = 42
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
    reg_weight: float,
) -> tuple[Scalar, Float64[Array, "receptors"], Scalar]:
    """
    JAX-optimized core optimization function using L-BFGS-B.
    """

    minAffs = jnp.full(targRecs_jax.shape[1], affinity_bounds[0])
    maxAffs = jnp.full(targRecs_jax.shape[1], affinity_bounds[1])

    # Start optimization at random values between min and max bounds
    # key = jax.random.PRNGKey(INIT_AFF_SEED)
    # key1, key2 = jax.random.split(key)
    # initAffs = jax.random.uniform(
    #     key1, shape=(targRecs_jax.shape[1],), minval=minAffs, maxval=maxAffs
    # )
    # initKx_star = jax.random.uniform(
    #     key2, minval=jnp.log10(Kx_star_bounds[0]), maxval=jnp.log10(Kx_star_bounds[1])
    # )
    # params = jnp.concatenate((initAffs, jnp.array([initKx_star])))
    params = jnp.concatenate([jnp.array([6.0]), jnp.full(targRecs_jax.shape[1] - 1, 7.0), jnp.array([ -9.0 ])])

    # Set optimization bounds
    bounds = (
        jnp.concatenate(
            [minAffs, jnp.array([jnp.log10(Kx_star_bounds[0])])]
        ),  # lower bounds
        jnp.concatenate(
            [maxAffs, jnp.array([jnp.log10(Kx_star_bounds[1])])]
        ),  # upper bounds
    )

    # Replace zero receptor counts with small epsilon to avoid instability
    targRecs_clean = jnp.where(targRecs_jax == 0, REC_COUNT_EPS, targRecs_jax)
    offTargRecs_clean = jnp.where(offTargRecs_jax == 0, REC_COUNT_EPS, offTargRecs_jax)

    # Set up L-BFGS-B solver from jaxopt
    solver = jaxopt.LBFGSB(fun=min_off_targ_selec, maxiter=max_iter, tol=tol)

    # Run optimization with bounds passed to run method
    result = solver.run(
        init_params=params,
        bounds=bounds,
        targRecs=targRecs_clean,
        offTargRecs=offTargRecs_clean,
        dose=dose,
        valencies=valencies_jax,
        reg_weight=reg_weight,
    )

    final_params = result.params
    final_loss = result.state.value

    return final_loss, final_params[:-1], jnp.power(10, final_params[-1])


def optimize_affs(
    targRecs: np.ndarray,
    offTargRecs: np.ndarray,
    dose: float,
    valencies: np.ndarray,
    affinity_bounds: tuple[float, float] = (6.0, 12.0),
    Kx_star_bounds: tuple[float, float] = (2.24e-15, 2.24e-9),
    max_iter: int = 1000,
    tol: float = 1e-5,
    reg_weight: float = 0.0,
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
        affinity_bounds: minimum and maximum optimization bounds for affinity values
        Kx_star_bounds: minimum and maximum optimization bounds for Kx_star
        max_iter: maximum number of iterations
        tol: parameter tolerance for convergence
        reg_weight: regularization weight for variance in target binding
        
    Outputs:
        optSelec: optimized selectivity value
        optAffs: optimized affinity values
        optKx_star: optimized Kx_star value
    """

    assert targRecs.size > 0
    assert offTargRecs.size > 0
    assert targRecs.shape[1] == offTargRecs.shape[1]

    # Convert inputs to JAX arrays
    targRecs_jax = jnp.array(targRecs, dtype=jnp.float64)
    dose_jax = jnp.array(dose, dtype=jnp.float64)
    offTargRecs_jax = jnp.array(offTargRecs, dtype=jnp.float64)
    valencies_jax = jnp.array(valencies, dtype=jnp.float64)

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
        reg_weight,
    )

    # Convert outputs back to Python types
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
    Kx_star: float = 2.24e-12,
    clean_zeros: bool = False,
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

    if clean_zeros:
        recCounts = jnp.where(recCounts == 0, REC_COUNT_EPS, recCounts)

    # Use the binding model to calculate bound receptors for each cell
    Rbound = cyt_binding_model_jit(
        dose=dose,
        recCounts=recCounts,
        valencies=valencies,
        monomerAffs=modelAffs,
        Kx_star=Kx_star,
    )
    
    return Rbound
