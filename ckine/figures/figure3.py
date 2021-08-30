"""
This creates Figure 1, response of bispecific IL-2 cytokines at varing valencies and abundances using binding model.
"""
from .figureCommon import getSetup, plotBispecific
from os.path import join
from ..imports import importCITE
import os

path_here = os.path.dirname(os.path.dirname(__file__))

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    ax, f = getSetup((15, 8), (2, 2))

    print(importCITE())

    return f
