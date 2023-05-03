"""
This creates Figure 5, used to find optimal epitope classifier.
"""
from os.path import dirname, join
from .common import getSetup
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares




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
    
    x = np.arange(0,100, 1)
    y = 2*x + 5
    noise = np.random.normal(loc=0, scale=1, size=len(x))
    y_noisy = y + noise

    def residuals(params, X, Y):
        A, B = params
        model_predictions = A*X + B
        error = model_predictions - Y
        return error
    
    ogparams = [0,0]
    optimized = least_squares(residuals, ogparams, args=(x, y_noisy))
    optimized_params = optimized.x
    ax[1].scatter(x, y_noisy, label='Simulated data')
    ax[1].plot(x, optimized_params[0]*x + optimized_params[1], 'r-', label='Fitted line')
    ax[1].legend()
    return f

   