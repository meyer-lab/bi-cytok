from os.path import dirname, join
from .common import getSetup
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from .common import Wass_KL_Dist
from .common import Wass_KL_Dist2d
from ..imports import importCITE


path_here = dirname(dirname(__file__))

def makeFigure():
    markerDF = importCITE()
    ax, f = getSetup((8, 8), (2, 2)) # works
    
    # Wass_KL_Dist(ax[2:4], targCell, numFactors) # works
    numFactors = 5
    receptor2 = ['CD122']
    receptor1 =['CD335'] # ??? or receptor1 = [receptor for receptor in markerDF.columns if receptor != "CD122"]
    targCell = "Treg Memory" # works 
    
    Wass_KL_Dist2d(ax[0:2], targCell, numFactors, receptor1, receptor2, RNA=False, offTargState=0)
    
    return f
