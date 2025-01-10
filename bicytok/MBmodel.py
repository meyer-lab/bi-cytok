"""
Implementation of a simple multivalent binding model.
"""

import numpy as np
from valentbind.model import polyc


def cytBindingModel(
    dose: float, # L0
    recCounts: np.ndarray, # Rtot
    valencies: np.ndarray, # Cplx
    monomerAffs: np.ndarray # Kav
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
        Lbound, Rbound, Lfbnd = polyc(
            L0 = ligandConc, 
            KxStar = 2.24e-12, 
            Rtot = Rtot, 
            Cplx = valencies, 
            Ctheta = np.full(len(valencies), 1/len(valencies)),
            Kav = monomerAffs
        )
        output = Rbound[0]

    # Calculate result for a multi-cell input (2D receptor counts)
    else:
        output = np.zeros_like(recCounts)
        for i in range(recCounts.shape[0]):
            Rtot = recCounts[i, :]
            Lbound, Rbound, Lfbnd = polyc(
                L0 = ligandConc, 
                KxStar = 2.24e-12, 
                Rtot = Rtot, 
                Cplx = valencies, 
                Ctheta = [1], 
                Kav = monomerAffs
            )
            output[i, :] = Rbound

    assert output.shape == recCounts.shape

    return output
