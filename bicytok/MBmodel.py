"""
Implementation of a simple multivalent binding model.
"""

import numpy as np
from .BindingMod import polyc
# from valentBind import polyc # Look into differences between valentBind and BindingMod versions of polyc


def cytBindingModel(
    recCount: np.ndarray, recXaffs: np.ndarray, dose: float, valencies: np.ndarray
) -> np.ndarray:
    """
    Runs binding model for a given mutein, valency, dose, and cell type
    Args:
        recCount: total count of signaling and targeting receptors
        recXaffs: Ka for monomer ligand to receptors
        dose: ligand concentration/dose that is being modeled
        valencies: array of valencies of each ligand epitope
    Return:
        output: amount of receptor bound of each kind of receptor
    """

    Kx = 2.24e-12
    ligandConc = dose / (valencies[0][0] * 1e9)

    output = polyc(ligandConc, Kx, recCount, valencies, recXaffs)[0]

    return output
