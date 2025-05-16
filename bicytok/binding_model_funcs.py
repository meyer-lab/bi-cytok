"""
Implementation of a simple multivalent binding model.
"""

import numpy as np
from scipy.optimize import fixed_point


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

    log_Req = np.full_like(Rtot, 0.0)

    for i in range(recCounts.shape[0]):
        log_Req[i] = fixed_point(
            Req_from_Req,
            x0=np.log(Rtot[i]) - 1.0,
            xtol=1e-6,
            args=(Cplxsum, L0_Ctheta_KxStar, Ka_KxStar, Rtot[i]),
        )

    return Rtot - np.exp(log_Req)


def Req_from_Req(
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
    return np.log(Rtot - Rbound)
