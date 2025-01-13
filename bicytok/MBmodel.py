"""
Implementation of a simple multivalent binding model.
"""

import numpy as np
from scipy.optimize import least_squares


def commonChecks(
    L0: float,
    Rtot: np.ndarray,
    KxStar: float,
    Kav: np.ndarray,
    Ctheta: np.ndarray,
):
    """Check that the inputs are sane."""
    Kav = np.array(Kav, dtype=float)
    Rtot = np.array(Rtot, dtype=float)
    Ctheta = np.array(Ctheta, dtype=float)
    assert Rtot.ndim <= 1
    assert Rtot.size == Kav.shape[1]
    assert Kav.ndim == 2
    assert Ctheta.ndim <= 1
    Ctheta = Ctheta / np.sum(Ctheta)
    return L0, Rtot, KxStar, Kav, Ctheta


def Req_polyc(
    Req, Rtot: np.ndarray, L0: float, KxStar, Cplx, Ctheta, Kav: np.ndarray
) -> np.ndarray:
    Psi = Req * Kav * KxStar
    Psirs = Psi.sum(axis=1).reshape(-1, 1) + 1
    Psinorm = Psi / Psirs

    Rbound = (
        L0
        / KxStar
        * np.sum(
            Ctheta.reshape(-1, 1)
            * np.dot(Cplx, Psinorm)
            * np.exp(np.dot(Cplx, np.log(Psirs))),
            axis=0,
        )
    )
    return Req + Rbound - Rtot


def polyc(
    L0: float,
    KxStar: float,
    Rtot: np.ndarray,
    Cplx: np.ndarray,
    Ctheta: np.ndarray,
    Kav: np.ndarray,
):
    """
    The main function to be called for multivalent binding
    :param L0: concentration of ligand complexes
    :param KxStar: Kx for detailed balance correction
    :param Rtot: numbers of each receptor on the cell
    :param Cplx: the monomer ligand composition of each complex
    :param Ctheta: the composition of complexes
    :param Kav: Ka for monomer ligand to receptors
    :return:
        Lbound: a list of Lbound of each complex
        Rbound: a list of Rbound of each kind of receptor
    """
    # Consistency check
    L0, Rtot, KxStar, Kav, Ctheta = commonChecks(L0, Rtot, KxStar, Kav, Ctheta)
    Cplx = np.array(Cplx)
    assert Cplx.ndim == 2
    assert Kav.shape[0] == Cplx.shape[1]
    assert Cplx.shape[0] == Ctheta.size

    # Solve Req
    lsq = least_squares(
        Req_polyc,
        np.zeros_like(Rtot),
        jac="cs",
        args=(Rtot, L0, KxStar, Cplx, Ctheta, Kav),
    )
    Req = lsq.x

    # Calculate the results
    Psi = Req.T * Kav * KxStar
    Psi = np.concatenate((Psi, np.ones((Kav.shape[0], 1))), axis=1)
    Psirs = np.sum(Psi, axis=1).reshape(-1, 1)
    Psinorm = (Psi / Psirs)[:, :-1]

    Rbound = (
        L0
        / KxStar
        * Ctheta.reshape(-1, 1)
        * np.dot(Cplx, Psinorm)
        * np.exp(np.dot(Cplx, np.log(Psirs)))
    )
    assert Rbound.shape[0] == len(Ctheta)
    assert Rbound.shape[1] == len(Rtot)
    return Rbound


def cytBindingModel(
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

    assert len(recCounts.shape) == 1 or len(recCounts.shape) == 2
    assert valencies[0].shape[0] == monomerAffs.shape[0]
    if len(recCounts.shape) == 1:
        assert recCounts.shape[0] == monomerAffs.shape[1]
    else:
        assert recCounts.shape[1] == monomerAffs.shape[1]

    # Armaan: Why 1e9? Again, it should be clear why literals are chosen.
    # Sam: valencies not always in nested vector form (see Fig1).
    #   What happens if valencies is a 1D vector?
    ligandConc = dose / (valencies[0][0] * 1e9)

    # Calculate result for a single cell input (1D receptor counts)
    if len(recCounts.shape) == 1:
        Rtot = recCounts
        Rbound = polyc(
            L0=ligandConc,
            KxStar=2.24e-12,
            Rtot=Rtot,
            Cplx=valencies,
            Ctheta=np.full(len(valencies), 1 / len(valencies)),
            Kav=monomerAffs,
        )
        output = Rbound[0]

    # Calculate result for a multi-cell input (2D receptor counts)
    else:
        output = np.zeros_like(recCounts)
        for i in range(recCounts.shape[0]):
            Rtot = recCounts[i, :]
            Rbound = polyc(
                L0=ligandConc,
                KxStar=2.24e-12,
                Rtot=Rtot,
                Cplx=valencies,
                Ctheta=np.array([1]),
                Kav=monomerAffs,
            )
            output[i, :] = Rbound

    assert output.shape == recCounts.shape

    return output
