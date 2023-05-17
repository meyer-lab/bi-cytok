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
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from ..BindingMod import polyc

path_here = dirname(dirname(__file__))


def makeFigure():
    ax, f = getSetup((12, 8), (3, 3))
    
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

    california_housing = fetch_california_housing()
    X, Y = california_housing.data, california_housing.target
    
    model = PLSRegression()
    model.fit(X,Y)
    
    Y_pred = model.predict(X)
    ax[2].scatter(Y, Y_pred, label='Actual values vs Predictions')

    x_loadings = model.x_loadings_[:, :2]
    y_loadings = model.y_loadings_[:, :2]
    feature_names = california_housing.feature_names

    for i in range(x_loadings.shape[0]):
        ax[3].annotate(feature_names[i], (x_loadings[i, 0], x_loadings[i, 1]))
    ax[3].set_xlabel('Component 1')
    ax[3].set_ylabel('Component 2')
    ax[3].set_title('X Loadings')

    ax[3].scatter(x_loadings[:, 0], x_loadings[:, 1])
   
    ax[4].scatter(y_loadings[:, 0], y_loadings[:, 1])
    ax[4].annotate('Price', (y_loadings[0, 0], y_loadings[0, 1]))
    ax[4].set_xlabel('Component 1')
    ax[4].set_ylabel('Component 2')
    ax[4].set_title('Y Loadings')

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    Y_true, Y_pred = [], []
    for train_index, test_index in kf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        model.fit(X_train, Y_train)
        Y_test_pred = model.predict(X_test)

        Y_true.extend(Y_test)
        Y_pred.extend(Y_test_pred)

    ax[5].scatter(Y_true, Y_pred, label='Actual values vs Predictions')
    ax[5].set_xlabel('Actual prices')
    ax[5].set_ylabel('Predicted prices')
    ax[5].set_title('Actual vs Predicted Prices')
    accuracy = r2_score(Y_true, Y_pred)
    print('R2 Accuracy:', accuracy)
    
    """ Slayboss model time begins here"""
    Kx = 1e-12
    Rtot_1 = [100]
    Rtot_2 = [10000]
    cplx_mono = [1]
    cplx_bi = [2]
    Ctheta = [1]
    Kav = 1e9
    conc_range = np.logspace(-12, -9, num=100)
    

    _, Rbound_mono_1, _ = polyc(conc_range, Kx, Rtot_1, cplx_mono, Ctheta, Kav)
    _, Rbound_mono_2, _ = polyc(conc_range, Kx, Rtot_2, cplx_mono, Ctheta, Kav)
    _, Rbound_bi_1, _ = polyc(conc_range, Kx, Rtot_1, cplx_bi, Ctheta, Kav)
    _, Rbound_bi_2, _ = polyc(conc_range, Kx, Rtot_2, cplx_bi, Ctheta, Kav)

    sns.lineplot(x=conc_range, y=Rbound_mono_1, label='Cell Type 1', ax=ax[6])
    sns.lineplot(x=conc_range, y=Rbound_mono_2, label='Cell Type 2', ax=ax[6])
    ax[6].set_xscale('log')
    ax[6].set_xlabel('Concentration (M)')
    ax[6].set_ylabel('Receptor Bound')
    ax[6].set_title('Monovalnt Ligand')
    ax[6].legend()

    sns.lineplot(x=conc_range, y=Rbound_bi_1, label='Cell Type 1', ax=ax[7])
    sns.lineplot(x=conc_range, y=Rbound_bi_2, label='Cell Type 2', ax=ax[7])
    ax[7].set_xscale('log')
    ax[7].set_xlabel('Concentratin (M)')
    ax[7].set_ylabel('Receptor Bound')
    ax[7].set_title('Bivalent Ligand')
    ax[7].legend()


    return f

   