"""
Implementation of a simple multivalent binding model.
"""

import numpy as np

from .BindingMod import polyc


def getKxStar():
    return 2.24e-12


def cytBindingModel(
    recCount: np.ndarray, recXaffs: np.ndarray, dose: float, vals: np.ndarray
):
    """Runs binding model for a given mutein, valency, dose, and cell type
    Args:
        recCount: total count of signaling and targeting receptors
        recXaffs: Ka for monomer ligand to receptors
        dose: ligand concentration/dose that is being modeled
        vals: array of valencies of each ligand epitope
    Return:
        output: amount of receptor bound of each kind of receptor
    """
    Kx = getKxStar()
    ligandConc = dose / (vals[0][0] * 1e9)

    output = polyc(ligandConc, Kx, recCount, vals, recXaffs)[0]

    return output
