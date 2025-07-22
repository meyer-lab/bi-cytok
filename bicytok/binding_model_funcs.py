"""
Implementation of a simple multivalent binding model.
"""

import jax
import jax.numpy as jnp
import optimistix as opt

jax.config.update("jax_enable_x64", True)


def cyt_binding_model(
    dose: float,
    recCounts: jnp.ndarray,
    valencies: jnp.ndarray,
    monomerAffs: jnp.ndarray,
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
        recCounts (np.ndarray): A 2D array where each row represents a
            system and each column represents the number of a specific
            receptor type in that system.
        valencies (np.ndarray): A 2D array (1 x number of complexes)
            representing the valency of each complex.
        monomerAffs (np.ndarray): A 2D array representing the affinity
            of each monomer for each receptor. Rows correspond to complexes,
            columns correspond to receptors.

    Returns:
        np.ndarray: A 2D array with the same shape as recCounts,
            representing the amount of each receptor bound to the ligand
            in each system.
    """
    L0, KxStar, Rtot, Cplx, Ctheta, Ka = reformat_parameters(
        dose, recCounts, valencies, monomerAffs
    )

    Rbound = infer_Rbound_batched_jax(
        L0,
        KxStar,
        Rtot,
        Cplx,
        Ctheta,
        Ka,
    )

    return Rbound


@jax.jit
def infer_Rbound_batched_jax(
    L0: jnp.ndarray,  # n_samples
    KxStar: jnp.ndarray,  # n_samples
    Rtot: jnp.ndarray,  # n_samples x n_R
    Cplx: jnp.ndarray,  # n_samples x n_cplx x n_L
    Ctheta: jnp.ndarray,  # n_samples x n_cplx
    Ka: jnp.ndarray,  # n_samples x n_L x n_R
) -> jnp.ndarray:
    def process_sample(i):
        return infer_Req(Rtot[i], L0[i], KxStar[i], Cplx[i], Ctheta[i], Ka[i])

    Req = jax.vmap(process_sample)(jnp.arange(Ka.shape[0]))
    return Rtot - Req


def infer_Req(
    Rtot: jnp.ndarray,
    L0: jnp.ndarray,
    KxStar: jnp.ndarray,
    Cplx: jnp.ndarray,
    Ctheta: jnp.ndarray,
    Ka: jnp.ndarray,
) -> jnp.ndarray:
    L0_Ctheta_KxStar = L0 * jnp.sum(Ctheta) / KxStar
    Ka_KxStar = Ka * KxStar
    Cplxsum = Cplx.sum(axis=0)

    def residual_log(log_Req: jnp.ndarray, _args) -> jnp.ndarray:
        Req = jnp.exp(log_Req)
        Rbound = infer_Rbound_from_Req(Req, Cplxsum, L0_Ctheta_KxStar, Ka_KxStar)
        return jnp.log(Rtot) - jnp.log(Req + Rbound)

    solver = opt.LevenbergMarquardt(rtol=1e-9, atol=1e-9)
    solution = opt.least_squares(
        residual_log,
        solver,
        y0=jnp.log(Rtot / 100.0),
        throw=False,
    )

    Req_opt = jnp.exp(solution.value)
    return Req_opt


def infer_Rbound_from_Req(
    Req: jnp.ndarray,
    Cplxsum: jnp.ndarray,
    L0_Ctheta_KxStar: jnp.ndarray,
    Ka_KxStar: jnp.ndarray,
) -> jnp.ndarray:
    Psi = Req * Ka_KxStar
    Psirs = Psi.sum(axis=1) + 1
    Psinorm = Psi / Psirs[:, None]
    return L0_Ctheta_KxStar * jnp.prod(Psirs**Cplxsum) * jnp.dot(Cplxsum, Psinorm)


def reformat_parameters(
    dose: float,
    recCounts: jnp.ndarray,
    valencies: jnp.ndarray,
    monomerAffs: jnp.ndarray,
    Kx_star: float = 2.24e-12,
) -> tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    """Reformats parameters to be compatible with the batched binding model."""

    assert recCounts.ndim == 2
    assert monomerAffs.ndim == 2
    assert valencies.ndim == 2
    assert monomerAffs.shape[0] == valencies.shape[1]
    assert valencies[0].shape[0] == monomerAffs.shape[0]
    assert recCounts.shape[1] == monomerAffs.shape[1]
    assert valencies.shape[0] == 1

    num_cells = recCounts.shape[0]
    num_receptors = recCounts.shape[1]

    ligand_conc = dose / (valencies[0][0] * 1e9)
    L0 = jnp.full(num_cells, ligand_conc)
    Kx_star_array = jnp.full(num_cells, 2.24e-12)
    Cplx = jnp.full((num_cells, 1, num_receptors), valencies)
    Ctheta = jnp.full((num_cells, 1), 1.0)
    Ka = jnp.full((num_cells, num_receptors, num_receptors), monomerAffs)

    assert L0.dtype == jnp.float64
    assert Kx_star_array.dtype == jnp.float64
    assert recCounts.dtype == jnp.float64
    assert Ka.dtype == jnp.float64
    assert Ctheta.dtype == jnp.float64
    assert L0.ndim == 1
    assert Kx_star_array.ndim == 1
    assert Ka.ndim == 3
    assert L0.shape[0] == Kx_star_array.shape[0]
    assert L0.shape[0] == recCounts.shape[0]
    assert Ctheta.shape == (L0.shape[0], Cplx.shape[1])
    assert Cplx.shape == (L0.shape[0], Ctheta.shape[1], Ka.shape[1])
    assert L0.shape[0] == Ka.shape[0]

    return L0, Kx_star_array, recCounts, Cplx, Ctheta, Ka
