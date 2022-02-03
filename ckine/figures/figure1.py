"""
This creates Figure 1, response of bispecific IL-2 cytokines at varing valencies and abundances using binding model.
"""

import numpy as np
from .figureCommon import getSetup, plotBispecific
from ..MBmodel import runFullModel_bispec


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    conc = np.array([1])

    modelDF = runFullModel_bispec(conc)

    print(modelDF)

    cells = ["Treg", "Thelper", "NK", "CD8"]
    ax, f = getSetup((15, 8), (2, 2))

    for i, cell in enumerate(cells):
        plotBispecific(ax[i], modelDF, cell)

    return f

# calc at different valencie
