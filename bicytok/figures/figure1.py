"""
This creates Figure 5, used to find optimal epitope classifier.
"""
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
from ..imports import importCITE


def makeFigure():
    ax, f = getSetup((12, 12), (1, 1))
    a = importCITE()
    sns.boxplot(data=a, x="CellType2", y="CD278", ax=ax[0])
    ax[0].set_xticklabels(labels=ax[0].get_xticklabels(), rotation=45, horizontalalignment='right')

    return f

   