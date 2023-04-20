"""
This creates Figure 5, used to find optimal epitope classifier.
"""
from os.path import dirname, join
from .common import getSetup
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


path_here = dirname(dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    ax, f = getSetup((8, 8), (2, 2))
    X = np.arange(0, 100)
    Y = np.arange(0, 100)
    ax[0].scatter(X, Y)
    ax[1].plot(X, Y)
    plotData = pd.DataFrame({"X": X, "Y": Y})
    sns.scatterplot(data=plotData, x="X", y="Y", ax=ax[2])
    sns.lineplot(data=plotData, x="X", y="Y", ax=ax[3])

    return f
