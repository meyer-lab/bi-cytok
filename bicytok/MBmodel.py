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
    """Runs binding model for a given mutein, valency, dose, and cell type."""
    # Check that values are in correct placement, can invert
    Kx = getKxStar()

    output = polyc(dose / (vals[0] * 1e9), Kx, recCount, [vals], recXaffs)[0]

    return output
