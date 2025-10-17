"""
Implementation of a simple multivalent binding model.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as opt

jax.config.update("jax_enable_x64", True)


def cyt_binding_model(
    dose: jnp.ndarray | float,
    recCounts: jnp.ndarray | np.ndarray,
    valencies: jnp.ndarray | np.ndarray,
    monomerAffs: jnp.ndarray | np.ndarray,
    Kx_star: float | jnp.ndarray = 2.24e-12,
) -> jnp.ndarray:
    """
    Calculate the amount of receptor bound to ligand at a given dose,
    considering receptor counts, valencies, and monomer affinities.

    This function models the binding of a ligand to receptors, taking into
    account the number of receptors, the valency of the ligand, and the
    affinity of the ligand for each receptor.  It assumes that each system
    has the same number of ligands, receptors, and complexes.

    Args:
        dose (float): The concentration of the ligand in molar units.
        recCounts (jnp.ndarray): A 2D array where each row represents a
            system and each column represents the number of a specific
            receptor type in that system.
        valencies (np.ndarray): A 2D array (1 x number of complexes)
            representing the valency of each complex.
        monomerAffs (np.ndarray): A 2D array representing the affinity
            of each monomer for each receptor. Rows correspond to complexes,
            columns correspond to receptors.
        Kx_star (float, optional): A float representing the cross-linking
            constant which describes all secondary binding events.

    Returns:
        np.ndarray: A 2D array with the same shape as recCounts,
            representing the amount of each receptor bound to the ligand
            in each system.
    """
    assert recCounts.ndim == 2
    assert monomerAffs.shape == (recCounts.shape[1], valencies.shape[1])

    infer_Req_vmap = jax.vmap(infer_Req, in_axes=(0, None, None, None, None))

    recCounts = jnp.array(recCounts)
    Req = infer_Req_vmap(recCounts, dose, Kx_star, valencies, monomerAffs)

    return recCounts - Req


def infer_Req(
    Rtot: jnp.ndarray,
    L0: jnp.ndarray | float,
    KxStar: jnp.ndarray | float,
    Cplx: jnp.ndarray,
    Ka: jnp.ndarray,
) -> jnp.ndarray:
    L0_KxStar = L0 / KxStar
    Ka_KxStar = Ka * KxStar
    Cplxsum = Cplx.sum(axis=0)

    def residual_log(log_Req: jnp.ndarray, _args) -> jnp.ndarray:
        """The polyc model from Tan et al."""
        Req = jnp.exp(log_Req)
        Psi = Req * Ka_KxStar
        Psirs = Psi.sum(axis=1) + 1
        Psinorm = Psi / Psirs[:, None]
        Rbound = L0_KxStar * jnp.prod(Psirs**Cplxsum) * jnp.dot(Cplxsum, Psinorm)
        return jnp.log(Rtot) - jnp.log(Req + Rbound)

    solver = opt.LevenbergMarquardt(rtol=1e-10, atol=1e-10)
    solution = opt.least_squares(
        residual_log,
        solver,
        y0=jnp.log(Rtot / 100.0),
        throw=False,
    )

    Req_opt = jnp.exp(solution.value)
    return Req_opt
