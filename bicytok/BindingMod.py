"""
Implementation of a simple multivalent binding model.
"""

import numpy as np
from numba import njit
from scipy.optimize import leastsq


@njit(parallel=False)
def Req_func2(
    Req: np.ndarray,
    Rtot: np.ndarray,
    L0: float,
    KxStar: float,
    Cplx: np.ndarray,
    Kav: np.ndarray,
):
    Psi = Req * Kav * KxStar
    Psirs = Psi.sum(axis=1) + 1
    Psinorm = Psi / Psirs[:, np.newaxis]

    Rbound = L0 / KxStar * np.exp(Cplx @ np.log(Psirs)) @ Cplx @ Psinorm
    return Req + Rbound - Rtot


def polyc(
    L0: float, KxStar: float, Rtot: np.ndarray, Cplx: list[np.ndarray], Kav: np.ndarray
):
    """
    The main function to be called for multivalent binding
    :param L0: concentration of ligand complexes
    :param KxStar: Kx for detailed balance correction
    :param Rtot: numbers of each receptor on the cell
    :param Cplx: the monomer ligand composition of each complex
    :param Kav: Ka for monomer ligand to receptors
    :return:
        Rtot - Req: amount of Rbound of each kind of receptor
    """
    # Consistency check
    Cplx = np.array(Cplx, dtype=float)
    assert Rtot.ndim <= 1
    assert Kav.shape == (Cplx.shape[1], Rtot.size)
    assert Cplx.ndim == 2
    assert Cplx.shape[0] == 1

    # Solve Req
    args = (Rtot, L0, KxStar, Cplx, Kav)
    Req, _, _, msg, ier = leastsq(
        Req_func2, np.zeros_like(Rtot), args=args, full_output=True
    )  # type: ignore
    assert ier in (1, 2, 3, 4), "Failure in rootfinding. " + str(msg)
    return Rtot - Req
