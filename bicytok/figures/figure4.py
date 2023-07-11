from os.path import dirname, join
from .common import getSetup
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from .common import EMD_Receptors
from .common import EMD_Distribution_Plot
from .common import EMD1D
from ..imports import importCITE


path_here = dirname(dirname(__file__))

def makeFigure():
    markerDF = importCITE()
    new_df = markerDF.head(1000)
    f, ax = plt.subplots(1, 4, figsize=(20, 5))
    target_cells = 'Treg'
    signaling_receptor = 'CD122'
    non_siganling_receptor = 'CD25'
    EMD_Receptors(new_df, signaling_receptor, target_cells, ax[0])
    EMD_Distribution_Plot(ax[1], new_df, signaling_receptor, non_siganling_receptor, target_cells)
    EMD1D(new_df, target_cells, ax[2])
    
    return f
