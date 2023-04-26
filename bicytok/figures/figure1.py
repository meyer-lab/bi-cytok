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
    ax, f = getSetup((8, 8), (2, 2))
    def linregression(params, Xs):
       A, B = params
       Ys = A*Xs + B
       return Ys
    def plotLin(Xs, Ys, ax):
        sns.lineplot(x=Xs, y=Ys, ax=ax)
    
    params = np.array([2,1])
    Xs = np.linspace(0,10, 100)
    Ys = linregression(params, Xs)
    plotLin(Xs, Ys, ax[0])
    return f

