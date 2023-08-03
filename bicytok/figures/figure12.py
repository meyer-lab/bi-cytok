from os.path import dirname, join
from .common import getSetup
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from .common import Wass_KL_Dist

path_here = dirname(dirname(__file__))

def makeFigure():
    ax, f = getSetup((8, 8), (3, 2))

    Wass_KL_Dist(ax[0:2], "Treg Memory", 10)
    Wass_KL_Dist(ax[2:4], "Treg Memory", 10, offTargState=1)
    Wass_KL_Dist(ax[4:6], "Treg Memory", 10, offTargState=2)
  
    return f
