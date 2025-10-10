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


# Define the function to minimize, which is the selectivity
min_off_targ_selec_jax = jax.jit(jax.value_and_grad(min_off_targ_selec))


@jax.jit
def optimize_affs(
    targRecs: np.ndarray,
    offTargRecs: np.ndarray,
    dose: float,
    valencies: np.ndarray,
    affinity_bounds: tuple[float, float] = (6.0, 12.0),
    Kx_star_bounds: tuple[float, float] = (2.24e-15, 2.24e-9),
    max_iter: int = 25,
    xtol: float = 1e-6,
    ftol: float = 1e-6,
    gtol: float = 1e-6,
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
        max_iter: maximum number of iterations
        xtol: parameter tolerance for convergence
        ftol: objective function tolerance for convergence
        gtol: gradient norm tolerance for convergence
    Return:
        optSelec: optimized selectivity value
        optAffs: optimized affinity values that yield the optimized selectivity
        optKx_star: optimized Kx_star value
    """

    assert targRecs.size > 0
    assert offTargRecs.size > 0

    # Convert inputs to JAX arrays
    targRecs = jnp.array(targRecs, dtype=jnp.float64)
    offTargRecs = jnp.array(offTargRecs, dtype=jnp.float64)
    dose = jnp.array(dose, dtype=jnp.float64)
    valencies = jnp.array(valencies, dtype=jnp.float64)

    minAffs = jnp.array([affinity_bounds[0]] * (targRecs.shape[1]))
    maxAffs = jnp.array([affinity_bounds[1]] * (targRecs.shape[1]))

    # Start optimization at random values between min and max bounds
    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)
    initAffs = jax.random.uniform(
        key1, shape=(len(minAffs),), minval=minAffs, maxval=maxAffs
    )
    initKx_star = jax.random.uniform(
        key2, minval=jnp.log10(Kx_star_bounds[0]), maxval=jnp.log10(Kx_star_bounds[1])
    )
    params = jnp.concatenate((initAffs, jnp.array([initKx_star])))

    # Set bounds for optimization
    minBounds = jnp.concatenate([minAffs, jnp.array([jnp.log10(Kx_star_bounds[0])])])
    maxBounds = jnp.concatenate([maxAffs, jnp.array([jnp.log10(Kx_star_bounds[1])])])

    targRecs = jnp.where(targRecs == 0, 1e-6, targRecs)
    offTargRecs = jnp.where(offTargRecs == 0, 1e-6, offTargRecs)

    solver = optax.lbfgs(
        linesearch=optax.scale_by_zoom_linesearch(
            max_linesearch_steps=55, verbose=False, initial_guess_strategy="one"
        )
    )
    opt_state = solver.init(params)

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

    def body_fn(loop_state):
        iteration, params, prev_params, prev_loss, loss, grads, opt_state = loop_state

        updates, new_opt_state = solver.update(
            grads,
            opt_state,
            params,
            value=loss,
            grad=grads,
            value_fn=min_off_targ_selec,
            targRecs=targRecs,
            offTargRecs=offTargRecs,
            dose=dose,
            valencies=valencies,
        )
        new_params = optax.apply_updates(params, updates)
        new_params = optax.projections.projection_box(new_params, minBounds, maxBounds)

        new_loss, new_grads = min_off_targ_selec_jax(
            new_params,
            targRecs,
            offTargRecs,
            dose,
            valencies,
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

    init_loss, init_grads = min_off_targ_selec_jax(
        params, targRecs, offTargRecs, dose, valencies
    )

    initial_state = (0, params, params, jnp.inf, init_loss, init_grads, opt_state)

    _, final_params, _, _, final_loss, _, _ = jax.lax.while_loop(
        cond_fn, body_fn, initial_state
    )

    return (
        final_loss.astype(float),
        list(final_params[:-1]),
        jnp.power(10, final_params[-1]).astype(float),
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
