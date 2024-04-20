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


def cytBindingModel(recCount: np.ndarray, holder: np.ndarray, dose: float, vals: np.ndarray, x=False):
    """Runs binding model for a given mutein, valency, dose, and cell type."""
    # Check that values are in correct placement, can invert
    doseVec = np.array(dose)
    Kx = getKxStar()

    if doseVec.size == 1:
        doseVec = np.array([doseVec])
    output = np.zeros(doseVec.size)

    for i, dose in enumerate(doseVec):
        output[i] = polyc(dose / (vals[0] * 1e9), Kx, recCount, [vals], holder)[0]

    return output
