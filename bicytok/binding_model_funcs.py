"""
Implementation of a simple multivalent binding model.
"""

import numpy as np
from scipy.optimize import least_squares


def _Rbound_from_Rbound(
    Rbound: np.ndarray,
    Cplxsum: np.ndarray,
    L0_Ctheta_KxStar: float,
    Ka_KxStar: np.ndarray,
    Rtot: np.ndarray,
) -> np.ndarray:
    Req = Rtot - Rbound
    Psi = Req * Ka_KxStar
    Psirs = Psi.sum(axis=1) + 1
    Psinorm = Psi / Psirs[:, None]
    Rbound = L0_Ctheta_KxStar * np.prod(Psirs**Cplxsum) * np.dot(Cplxsum, Psinorm)
    return Rtot - Rbound - Req


def cyt_binding_model(
    dose: float,
    recCounts: np.ndarray,
    valencies: np.ndarray,
    monomerAffs: np.ndarray,
) -> np.ndarray:
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

    Rbound = np.full_like(Rtot, 0.0)

    for i in range(recCounts.shape[0]):
        opt = least_squares(
            _Rbound_from_Rbound,
            x0=Rtot[i] / 2.0,
            xtol=1e-12,
            gtol=1e-12,
            jac="cs",
            args=(Cplxsum, L0_Ctheta_KxStar, Ka_KxStar, Rtot[i]),
        )
        assert opt.cost < 1.0e-6
        assert opt.success
        Rbound[i] = opt.x

    return Rbound
