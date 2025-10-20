"""
Implementation of a simple multivalent binding model.
"""

import jax
import jax.numpy as jnp
import optimistix as opt
from jaxtyping import Array, Float64, Scalar

jax.config.update("jax_enable_x64", True)


def cyt_binding_model(
    dose: Scalar,
    recCounts: Float64[Array, "cells receptors"],  # type: ignore
    valencies: Float64[Array, "receptors"],  # type: ignore
    monomerAffs: Float64[Array, "receptors receptors"],  # type: ignore
    Kx_star: Scalar,
) -> Float64[Array, "cells receptors"]:  # type: ignore
    """
    Calculate the amount of receptor bound to ligand at a given dose,
    considering receptor counts, valencies, and monomer affinities.

    This function models the binding of a ligand to receptors, taking into
    account the number of receptors, the valency of the ligand, and the
    affinity of the ligand for each receptor.  It assumes that each system
    has the same number of ligands, receptors, and complexes.

    Args:
        dose: The concentration of the ligand complex in molar units.
        recCounts: Receptor counts (columns) across cells (rows).
        valencies: The valency of each ligand complex (just one distinct complex for
            our purposes).
        monomerAffs: The affinity of each ligand monomer for each receptor.
        Kx_star: The cross-linking constant which describes all secondary binding
            events.

    Returns:
        Rbound: The amount of each receptor bound to each ligand on each cell.
    """
    assert recCounts.ndim == 2
    assert monomerAffs.shape == (recCounts.shape[1], valencies.shape[1])

    infer_Req_vmap = jax.vmap(infer_Req, in_axes=(0, None, None, None, None))

    recCounts = jnp.array(recCounts)
    Req = infer_Req_vmap(recCounts, dose, Kx_star, valencies, monomerAffs)

    return recCounts - Req


def infer_Req(
    Rtot: Float64[Array, "receptors"],  # type: ignore
    L0: Scalar,
    KxStar: Scalar,
    Cplx: Float64[Array, "receptors"],  # type: ignore
    Ka: Float64[Array, "receptors receptors"],  # type: ignore
) -> Float64[Array, "receptors"]:  # type: ignore
    L0_KxStar = L0 / KxStar
    Ka_KxStar = Ka * KxStar
    Cplxsum = Cplx.sum(axis=0)

    def residual_log(
        log_Req: Float64[Array, "receptors"], _args
    ) -> Float64[Array, "receptors"]:  # type: ignore
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
