"""
This creates Figure 5, used to find optimal epitope classifier.
"""
from os.path import dirname, join
from .common import getSetup
import pandas as pd
import seaborn as sns
import numpy as np
import scipy

from valentbind import polyc

path_here = dirname(dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    ax, f = getSetup((12, 9), (1, 1))

    doseVec = np.logspace(-12, -6, num=100)
    vals = [1, 2, 3, 4]

    output = np.zeros(doseVec.size)

    df = pd.DataFrame(columns=['Valency', 'Dose', 'Receptor Bound'])

    for val in vals:
        for i, dose in enumerate(doseVec):
            output[i] = polyc(dose, 1e-12, [10000], [[val]], [1.0], np.array([[1e9]]))[0][1]
        data = {'Valency': val,
            'Dose': doseVec,
            'Receptor Bound': output
        }
        df2 = pd.DataFrame(data, columns=['Valency', 'Dose', 'Receptor Bound'])
        df = df.append(df2, ignore_index=True)

    sns.lineplot(data=df, x='Dose', y='Receptor Bound', hue='Valency', ax=ax[0])
    ax[0].set(xscale='log')

    def residuals(x):
        targetbinding = polyc(1e-9, 1e-12, [1000, 1000], [[1]], [1.0], np.array([[1e9, np.power(10, x[0])]]))[0][1]
        offtargetbinding = polyc(1e-9, 1e-12, [1000, 100], [[1]], [1.0], np.array([[1e9, np.power(10, x[0])]]))[0][1]
        return offtargetbinding / targetbinding

    def minimizefunction():
        fun = lambda x: residuals(x)
        x0 = np.log10(1e9)
        bounds = [(7, 10)]
        minimized = scipy.optimize.minimize(fun, x0, bounds=bounds)
        return minimized.x

    print(minimizefunction())

    return f
