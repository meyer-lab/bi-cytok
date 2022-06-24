"""
This creates Figure 5, used to find optimal epitope classifier.
"""
from os.path import dirname, join
from .common import getSetup
import pandas as pd
import seaborn as sns
import numpy as np

from valentbind import polyc

path_here = dirname(dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    ax, f = getSetup((9, 12), (1, 1))

    doseVec = np.logspace(-12, -6, num=100)
    vals = [1, 2, 3, 4]

    output = np.zeros(doseVec.size)

    for val in vals:
        for i, dose in enumerate(doseVec):
            output[i] = polyc(dose, 1e-12, [10000], [[val]], [1.0], np.array([[1e9]]))[0][1]

    return f
