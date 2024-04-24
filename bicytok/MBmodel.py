"""
Implementation of a simple multivalent binding model.
"""

import os
import numpy as np
import pandas as pd
from .BindingMod import polyc
from .imports import getBindDict, importReceptors
from os.path import join

path_here = os.path.dirname(os.path.dirname(__file__))


def getKxStar():
    return 2.24e-12


def cytBindingModel(recCount: np.ndarray, recXaffs: np.ndarray, dose: float, vals: np.ndarray):
    """Runs binding model for a given mutein, valency, dose, and cell type."""
    # Check that values are in correct placement, can invert
    Kx = getKxStar()

    affs = pd.DataFrame()
    for i, recXaff in enumerate(recXaffs):
        affs = np.append(affs, np.power(10, recXaff))
    holder = np.full((recXaffs.size, recXaffs.size), 1e2)
    np.fill_diagonal(holder, affs)
    affs = holder

    output = polyc(dose / (vals[0] * 1e9), Kx, recCount, [vals], affs)[0]

    return output
