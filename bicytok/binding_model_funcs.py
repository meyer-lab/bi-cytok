"""
Implementation of a simple multivalent binding model.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jaxopt import LevenbergMarquardt

jax.config.update("jax_enable_x64", True)


def cyt_binding_model(
    dose: float,
    recCounts: np.ndarray,
    valencies: np.ndarray,
    monomerAffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Each system should have the same number of ligands, receptors, and complexes.
    Returns both bound receptors and optimization losses for each cell.
    """
    L0, KxStar, Rtot, Cplx, Ctheta, Ka = reformat_parameters(
        dose, recCounts, valencies, monomerAffs
    )

    Rbound, losses = infer_Rbound_batched_jax(
        jnp.array(L0, dtype=jnp.float64),
        jnp.array(KxStar, dtype=jnp.float64),
        jnp.array(Rtot, dtype=jnp.float64),
        jnp.array(Cplx, dtype=jnp.float64),
        jnp.array(Ctheta, dtype=jnp.float64),
        jnp.array(Ka, dtype=jnp.float64),
    )

    return np.array(Rbound), np.array(losses)


@jax.jit
def infer_Rbound_batched_jax(
    L0: jnp.ndarray,  # n_samples
    KxStar: jnp.ndarray,  # n_samples
    Rtot: jnp.ndarray,  # n_samples x n_R
    Cplx: jnp.ndarray,  # n_samples x n_cplx x n_L
    Ctheta: jnp.ndarray,  # n_samples x n_cplx
    Ka: jnp.ndarray,  # n_samples x n_L x n_R
) -> tuple[jnp.ndarray, jnp.ndarray]:
    def process_sample(i):
        return infer_Req(Rtot[i], L0[i], KxStar[i], Cplx[i], Ctheta[i], Ka[i])

    Req, losses = jax.vmap(process_sample)(jnp.arange(Ka.shape[0]))

    _ = jax.lax.cond(
        jnp.any(losses > 1e-3),
        lambda _: jax.debug.print(
            "Losses exceeding threshold: {}", jnp.sum(losses > 1e-3)
        ),
        lambda _: None,
        operand=None,
    )

    return Rtot - Req, losses


def infer_Req(
    Rtot: jnp.ndarray,
    L0: jnp.ndarray,
    KxStar: jnp.ndarray,
    Cplx: jnp.ndarray,
    Ctheta: jnp.ndarray,
    Ka: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    L0_Ctheta_KxStar = L0 * jnp.sum(Ctheta) / KxStar
    Ka_KxStar = Ka * KxStar
    Cplxsum = Cplx.sum(axis=0)

    def residual_log(log_Req: jnp.ndarray) -> jnp.ndarray:
        Req = jnp.exp(log_Req)
        Rbound = infer_Rbound_from_Req(Req, Cplxsum, L0_Ctheta_KxStar, Ka_KxStar)
        return jnp.log(Rtot) - jnp.log(Req + Rbound)

    solver = LevenbergMarquardt(
        residual_log,
        damping_parameter=5e-4,
        maxiter=75,
        tol=1e-14,
        xtol=1e-14,
        gtol=1e-14,
        implicit_diff=False,
        jit=True,
    )
    solution = solver.run(jnp.log(Rtot / 100.0))
    Req_opt = jnp.exp(solution.params)

    loss = solution.state.value**2
    return Req_opt, loss


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
    recCounts: np.ndarray,
    valencies: np.ndarray,
    monomerAffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    L0 = np.full(num_cells, ligand_conc)
    Kx_star = np.full(num_cells, 2.24e-12)
    Cplx = np.full((num_cells, 1, num_receptors), valencies)
    Ctheta = np.full((num_cells, 1), 1.0)
    Ka = np.full((num_cells, num_receptors, num_receptors), monomerAffs)

    assert L0.dtype == np.float64
    assert Kx_star.dtype == np.float64
    assert recCounts.dtype == np.float64
    assert Ka.dtype == np.float64
    assert Ctheta.dtype == np.float64
    assert L0.ndim == 1
    assert Kx_star.ndim == 1
    assert Ka.ndim == 3
    assert L0.shape[0] == Kx_star.shape[0]
    assert L0.shape[0] == recCounts.shape[0]
    assert Ctheta.shape == (L0.shape[0], Cplx.shape[1])
    assert Cplx.shape == (L0.shape[0], Ctheta.shape[1], Ka.shape[1])
    assert L0.shape[0] == Ka.shape[0]

    return L0, Kx_star, recCounts, Cplx, Ctheta, Ka
