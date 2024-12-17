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
    Runs binding model for a given mutein, valency, dose, and cell type
    Args:
        dose: ligand concentration/dose that is being modeled
        recCounts: counts of each receptor on all single cells
        valencies: valencies of each ligand
        monomerAffs: binding affinities for monomer ligands to receptors 
    Return:
        output: counts of bound receptors on all single cells
    """

    # Armaan: Why 1e9? Again, it should be clear why literals are chosen.
    # Sam: valencies not always in nested vector form (see Fig1). What happens if valencies is a 1D vector?
    ligandConc = dose / (valencies[0][0] * 1e9)

    Lbound, Rbound = polyc(
        L0 = ligandConc, 
        KxStar = 2.24e-12, 
        Rtot = recCounts, 
        Cplx = valencies, 
        Ctheta = [1], 
        Kav = monomerAffs
    )

    return Rbound
