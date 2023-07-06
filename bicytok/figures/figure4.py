from os.path import dirname, join
from .common import getSetup
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from .common import Wass_KL_Dist
from .common import calculate_distance
from ..imports import importCITE


path_here = dirname(dirname(__file__))

def makeFigure():
    markerDF = importCITE()
    new_df = markerDF.head(1000)
    ax, f = getSetup((8, 8), (1, 1))
    target_cells = 'Treg'
    signaling_receptor = 'CD122'
    calculate_distance(new_df, signaling_receptor, target_cells, ax)
    
    return f
