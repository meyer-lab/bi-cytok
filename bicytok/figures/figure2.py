"""
This creates Figure 2, a practice assignment.
"""
from os.path import dirname, join
from .common import getSetup
import pandas as pd
import seaborn as sns
import numpy as np
import scipy

from valentbind import polyc
from ..imports import importReceptors
from ..MBmodel import getKxStar

path_here = dirname(dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    ax, f = getSetup((12, 9), (1, 1))

    def recCount(cellType):
        recDF = importReceptors()
        recCount = np.ravel(
            [
                recDF.loc[(recDF.Receptor == 'IL2Ra') & (recDF['Cell Type'] == cellType)].Mean.values,
                recDF.loc[(recDF.Receptor == 'IL2Rb') & (recDF['Cell Type'] == cellType)].Mean.values,
            ]
        )
        return recCount

    def residuals(x, val):
        nkbinding = polyc(1e-9, getKxStar(), recCount('NK'), [[val, val]], [1.0], np.array([[np.power(10, x[0]), 10], [10, np.power(10, x[1])]]))[0][1]
        tregbinding = polyc(1e-9, getKxStar(), recCount('Treg'), [[val, val]], [1.0], np.array([[np.power(10, x[0]), 10], [10, np.power(10, x[1])]]))[0][1]
        return nkbinding / tregbinding

    def minimizefunction(val):
        fun = lambda x: residuals(x, val)
        x0 = [[np.log10(1e9), np.log10(1e9)]]
        bounds = [(7, 10), (7, 10)]
        minimized = scipy.optimize.minimize(fun, x0, bounds=bounds)
        return minimized.x

    doseVec = np.logspace(-12, -6, num=100)
    vals = [1, 2, 3, 4]

    Treg = np.zeros(doseVec.size)
    NK = np.zeros(doseVec.size)
    output = np.zeros(doseVec.size)

    df = pd.DataFrame(columns=['Valency', 'Dose', 'Receptor Bound to Treg / Receptor Bound to NK cell'])

    for val in vals:
        affs = minimizefunction(val)
        for i, dose in enumerate(doseVec):
            Treg[i] = polyc(1e-9, getKxStar(), recCount('Treg'), [[val, val]], [1.0], np.array([[affs[0], 10], [10, affs[1]]]))[0][1]
            NK[i] = polyc(1e-9, getKxStar(), recCount('NK'), [[val, val]], [1.0], np.array([[affs[0], 10], [10, affs[1]]]))[0][1]
            output[i] = Treg[i] / NK[i]
        data = {'Valency': val,
            'Dose': doseVec,
            'Receptor Bound to Treg / Receptor Bound to NK cell': output
        }
        df2 = pd.DataFrame(data, columns=['Valency', 'Dose', 'Receptor Bound to Treg / Receptor Bound to NK cell'])
        df = df.append(df2, ignore_index=True)

    sns.lineplot(data=df, x='Dose', y='Receptor Bound to Treg / Receptor Bound to NK cell', hue='Valency', ax=ax[0])
    ax[0].set(xscale='log')

    return f
