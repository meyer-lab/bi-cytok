"""
Implementation of a simple multivalent binding model.
"""

import numpy as np
from scipy.optimize import least_squares


def cyt_binding_model(
    dose: float,
    recCounts: np.ndarray,
    valencies: np.ndarray,
    monomerAffs: np.ndarray,
) -> np.ndarray:
    """
    Each system should have the same number of ligands, receptors, and complexes.
    """
    assert recCounts.ndim == 2
    assert monomerAffs.ndim == 2
    assert valencies.ndim == 2
    assert monomerAffs.shape[0] == valencies.shape[1]
    assert valencies[0].shape[0] == monomerAffs.shape[0]
    assert recCounts.shape[1] == monomerAffs.shape[1]
    assert valencies.shape[0] == 1

    L0 = dose / (valencies[0][0] * 1e9)
    KxStar = 2.24e-12
    L0_Ctheta_KxStar = float(L0 / KxStar)
    Ka_KxStar = monomerAffs * KxStar
    Rtot = recCounts
    Cplxsum = valencies.sum(axis=0)

    Req = np.full_like(Rtot, 0.0)

    for i in range(recCounts.shape[0]):
        Req[i] = infer_Req(Rtot[i], L0_Ctheta_KxStar, Ka_KxStar, Cplxsum)

    return Rtot - Req


def infer_Rbound_from_Req(
    log_Req: np.ndarray,
    Cplxsum: np.ndarray,
    L0_Ctheta_KxStar: float,
    Ka_KxStar: np.ndarray,
    Rtot: np.ndarray,
) -> np.ndarray:
    Req = np.exp(log_Req)
    Psi = Req * Ka_KxStar
    Psirs = Psi.sum(axis=1) + 1
    Psinorm = Psi / Psirs[:, None]
    Rbound = L0_Ctheta_KxStar * np.prod(Psirs**Cplxsum) * np.dot(Cplxsum, Psinorm)
    return np.log(Rtot) - np.log(Req + Rbound)



def infer_Req(
    Rtot: np.ndarray,
    L0_Ctheta_KxStar: float,
    Ka_KxStar: np.ndarray,
    Cplxsum: np.ndarray,
) -> np.ndarray:
    sol = least_squares(
        infer_Rbound_from_Req,
        x0=Rtot - 2.0,
        method="lm",
        xtol=1e-14,
        args=(Cplxsum, L0_Ctheta_KxStar, Ka_KxStar, Rtot),
    )
    print(sol)

    assert np.linalg.norm(sol.fun) < 1.0e-6
    return np.exp(sol.x)
