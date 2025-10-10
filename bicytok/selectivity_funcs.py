"""
Functions used in binding and selectivity analysis
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float64, Scalar

from .binding_model_funcs import cyt_binding_model

jax.config.update("jax_enable_x64", True)
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".05"


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
    targRecs: Float64[Array, "cells receptors"],
    offTargRecs: Float64[Array, "cells receptors"],
    dose: Scalar,
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
    targetBound = jnp.sum(targRbound[:, 0]) / targRbound.shape[0]
    offTargetBound = jnp.sum(offTargRbound[:, 0]) / offTargRbound.shape[0]

    # Return selectivity ratio
    return (targetBound + offTargetBound) / targetBound


# Define the JAX-optimized objective function and its gradient (minimizing selectivity)
min_off_targ_selec_jax = jax.jit(jax.value_and_grad(min_off_targ_selec))

# Affinity optimization constants
INIT_AFF_SEED = 42
REC_COUNT_EPS = 1e-6


@jax.jit
def _optimize_affs_jax(
    targRecs_jax: Float64[Array, "cells receptors"],
    offTargRecs_jax: Float64[Array, "cells receptors"],
    dose: float,
    valencies_jax: Float64[Array, "complexes epitopes"],
    affinity_bounds: tuple[float, float],
    Kx_star_bounds: tuple[float, float],
    max_iter: int,
    xtol: float,
    ftol: float,
    gtol: float,
) -> tuple[Scalar, Float64[Array, "receptors"], Scalar]:
    """
    JAX-optimized core optimization function.
    """

    minAffs = jnp.full(targRecs_jax.shape[1], affinity_bounds[0])
    maxAffs = jnp.full(targRecs_jax.shape[1], affinity_bounds[1])

    # Start optimization at random values between min and max bounds
    key = jax.random.PRNGKey(INIT_AFF_SEED)
    key1, key2 = jax.random.split(key)
    initAffs = jax.random.uniform(
        key1, shape=(targRecs_jax.shape[1],), minval=minAffs, maxval=maxAffs
    )
    initKx_star = jax.random.uniform(
        key2, minval=jnp.log10(Kx_star_bounds[0]), maxval=jnp.log10(Kx_star_bounds[1])
    )
    params = jnp.concatenate((initAffs, jnp.array([initKx_star])))

    # Set bounds for optimization
    minBounds = jnp.concatenate([minAffs, jnp.array([jnp.log10(Kx_star_bounds[0])])])
    maxBounds = jnp.concatenate([maxAffs, jnp.array([jnp.log10(Kx_star_bounds[1])])])

    # Replace zero receptor counts with small epsilon to avoid instability
    targRecs_clean = jnp.where(targRecs_jax == 0, REC_COUNT_EPS, targRecs_jax)
    offTargRecs_clean = jnp.where(offTargRecs_jax == 0, REC_COUNT_EPS, offTargRecs_jax)

    # Set up the L-BFGS solver from Optax
    # Optax recommends max 55 linesearch steps for 64-bit precision
    solver = optax.lbfgs(
        linesearch=optax.scale_by_zoom_linesearch(
            max_linesearch_steps=55, verbose=False, initial_guess_strategy="one"
        )
    )
    opt_state = solver.init(params)

    # Condition function for jax while loop checks convergence criteria
    def cond_fn(loop_state):
        iteration, params, prev_params, prev_loss, loss, grads, opt_state = loop_state

        param_change = jnp.sqrt(jnp.sum((params - prev_params) ** 2))
        loss_change = jnp.abs(loss - prev_loss)
        grad_norm = jnp.sqrt(jnp.sum(grads**2))

        x_converged = param_change < xtol
        f_converged = loss_change < ftol
        g_converged = grad_norm < gtol
        converged = x_converged | f_converged | g_converged

        return (~converged) & (iteration < max_iter)

    # Body function for jax while loop performs one optimization step
    def body_fn(loop_state):
        iteration, params, prev_params, prev_loss, loss, grads, opt_state = loop_state

        updates, new_opt_state = solver.update(
            grads,
            opt_state,
            params,
            value=loss,
            grad=grads,
            value_fn=min_off_targ_selec,
            targRecs=targRecs_clean,
            offTargRecs=offTargRecs_clean,
            dose=dose,
            valencies=valencies_jax,
        )
        new_params = optax.apply_updates(params, updates)
        new_params = optax.projections.projection_box(new_params, minBounds, maxBounds)

        new_loss, new_grads = min_off_targ_selec_jax(
            new_params,
            targRecs_clean,
            offTargRecs_clean,
            dose,
            valencies_jax,
        )

        return (
            iteration + 1,
            new_params,
            params,
            loss,
            new_loss,
            new_grads,
            new_opt_state,
        )

    # Run the jax optimization loop
    init_loss, init_grads = min_off_targ_selec_jax(
        params, targRecs_clean, offTargRecs_clean, dose, valencies_jax
    )
    initial_state = (0, params, params, jnp.inf, init_loss, init_grads, opt_state)
    _, final_params, _, _, final_loss, _, _ = jax.lax.while_loop(
        cond_fn, body_fn, initial_state
    )

    return final_loss, final_params[:-1], jnp.power(10, final_params[-1])


def optimize_affs(
    targRecs: np.ndarray,
    offTargRecs: np.ndarray,
    dose: float,
    valencies: np.ndarray,
    affinity_bounds: tuple[float, float] = (6.0, 12.0),
    Kx_star_bounds: tuple[float, float] = (2.24e-15, 2.24e-9),
    max_iter: int = 100,
    xtol: float = 1e-12,
    ftol: float = 1e-12,
    gtol: float = 1e-12,
) -> tuple[float, list, float]:
    """
    NumPy-compatible wrapper for optimize_affs that handles conversions to/from JAX.
    Minimizes the off-target to on-target selectivity ratio by optimizing
    receptor affinities and Kx_star using L-BFGS.

    Args:
        targRecs: receptor counts of target cell type (NumPy array)
        offTargRecs: receptor counts of off-target cell types (NumPy array)
        dose: ligand concentration/dose that is being modeled
        valencies: array of valencies of each ligand epitope (NumPy array)
        affinity_bounds: minimum and maximum optimization bounds for affinity values
        Kx_star_bounds: minimum and maximum optimization bounds for Kx_star
        max_iter: maximum number of iterations
        xtol: parameter tolerance for convergence
        ftol: objective function tolerance for convergence
        gtol: gradient norm tolerance for convergence

    Return:
        optSelec: optimized selectivity value (Python float)
        optAffs: optimized affinity values (Python list)
        optKx_star: optimized Kx_star value (Python float)
    """

    assert targRecs.size > 0
    assert offTargRecs.size > 0
    assert targRecs.shape[1] == offTargRecs.shape[1]

    # Convert inputs to JAX arrays
    targRecs_jax = jnp.array(targRecs, dtype=jnp.float64)
    offTargRecs_jax = jnp.array(offTargRecs, dtype=jnp.float64)
    valencies_jax = jnp.array(valencies, dtype=jnp.float64)

    # Call the JAX-optimized function
    final_loss, final_affs, final_kx_star = _optimize_affs_jax(
        targRecs_jax,
        offTargRecs_jax,
        dose,
        valencies_jax,
        affinity_bounds,
        Kx_star_bounds,
        max_iter,
        xtol,
        ftol,
        gtol,
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

    monomerAffs = jnp.array(monomerAffs, dtype=jnp.float64)
    recCounts = jnp.array(recCounts, dtype=jnp.float64)
    valencies = jnp.array(valencies, dtype=jnp.float64)

    # Reformat input affinities to 10^aff and diagonalize
    modelAffs = restructure_affs(monomerAffs)

    # Use the binding model to calculate bound receptors for each cell
    Rbound = cyt_binding_model_jit(
        dose=dose,
        recCounts=recCounts,
        valencies=valencies,
        monomerAffs=modelAffs,
        Kx_star=Kx_star,
    )

    return Rbound
