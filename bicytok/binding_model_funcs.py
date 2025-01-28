"""
Implementation of a simple multivalent binding model.
"""

import numpy as np
from scipy.optimize import root


def Rbnd_polyc(
    Req, L0: float, KxStar, Cplx, Kav: np.ndarray
) -> np.ndarray:
    Psi = Req * Kav * KxStar
    Psirs = Psi.sum(axis=1) + 1
    Psinorm = Psi / Psirs[None, :]

    Rbound = (
        L0
        / KxStar
        * (Cplx @ Psinorm)
        * np.exp(Cplx @ np.log(Psirs))
    )
    return Rbound


def Req_polyc(
    Req, Rtot: np.ndarray, L0: float, KxStar, Cplx, Kav: np.ndarray
) -> np.ndarray:
    Rbound = Rbnd_polyc(Req, L0, KxStar, Cplx, Kav)
    return Rtot - Req - Rbound.sum(axis=1)


def polyc(
    L0: float,
    KxStar: float,
    Rtot: np.ndarray,
    Cplx: np.ndarray,
    Kav: np.ndarray,
):
    """
    The main function to be called for multivalent binding
    :param L0: concentration of ligand complexes
    :param KxStar: Kx for detailed balance correction
    :param Rtot: numbers of each receptor on the cell
    :param Cplx: the monomer ligand composition of each complex
    :param Kav: Ka for monomer ligand to receptors
    :return:
        Rbound: a list of Rbound of each kind of receptor
    """
    # Solve Req
    sol = root(
        Req_polyc,
        Rtot / 10.0,
        args=(Rtot, L0, KxStar, Cplx, Kav),
        method="lm",
    )
    Req = sol.x
    print(sol)
    if sol.status < 1:
        print(sol.message)

    return Rbnd_polyc(Req, L0, KxStar, Cplx, Kav)


def cyt_binding_model(
    dose: float,  # L0
    recCounts: np.ndarray,  # Rtot
    valencies: np.ndarray,  # Cplx
    monomerAffs: np.ndarray,  # Kav
) -> np.ndarray:
    """
    Runs binding model to calculate bound receptors
    for multiple receptor types on multiple single cells.
    Args:
        dose: ligand concentration/dose that is being modeled
        recCounts: counts of each receptor on all single cells
        valencies: valencies of each ligand
            a nested vector
        monomerAffs: binding affinities for monomer ligands to receptors
    Return:
        output: counts of bound receptors on all single cells
    """
    # Consistency check
    assert recCounts.ndim == 2
    assert monomerAffs.ndim == 2
    assert valencies.ndim == 2
    assert monomerAffs.shape[0] == valencies.shape[1]
    assert valencies[0].shape[0] == monomerAffs.shape[0]
    assert recCounts.shape[1] == monomerAffs.shape[1]
    assert valencies.shape[0] == 1

    # Needed for numba
    valencies = np.array(valencies, dtype=float)

    # Armaan: Why 1e9? Again, it should be clear why literals are chosen.
    # Sam: valencies not always in nested vector form (see Fig1).
    #   What happens if valencies is a 1D vector?
    ligandConc = dose / (valencies[0][0] * 1e9)

    output = np.zeros_like(recCounts)
    for i in range(recCounts.shape[0]):
        Rtot = recCounts[i, :]
        Rbound = polyc(
            L0=ligandConc,
            KxStar=2.24e-12,
            Rtot=Rtot,
            Cplx=valencies,
            Kav=monomerAffs,
        )
        output[i, :] = Rbound

    return output
