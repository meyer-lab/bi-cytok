"""
Implementation of a simple multivalent binding model.
"""

import numpy as np
from .BindingMod import polyc


def getKxStar():
    return 2.24e-12


def cytBindingModel(
    recCount: np.ndarray, recXaffs: np.ndarray, dose: float, vals: list
):
    """Runs binding model for a given mutein, valency, dose, and cell type."""
    # Check that values are in correct placement, can invert
    Kx = getKxStar()
    ligandConc = dose / (vals[0] * 1e9)

    output = polyc(ligandConc, Kx, recCount, np.array([vals]), recXaffs)[0]

    return output
