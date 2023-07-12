from os.path import dirname, join
from .common import getSetup
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from .common import EMD_2D
from .common import EMD_Distribution_Plot
from .common import EMD_1D
from .common import EMD1Dvs2D_Analysis
from ..imports import importCITE



path_here = dirname(dirname(__file__))

def makeFigure(): 
    markerDF = importCITE()
    new_df = markerDF.head(1000) 
    f, ax = plt.subplots(1, 4, figsize=(20, 5))   
    target_cells = 'Treg'
    signaling_receptor = 'CD122'
    non_siganling_receptor = 'CD25'
    receptor_names = ['CD4-1', 'CD27', 'CD3-2', 'CD109', 'CD278']
    # EMD_2D(new_df, signaling_receptor, target_cells, ax[0])
    EMD_Distribution_Plot(ax[2], new_df, signaling_receptor, non_siganling_receptor, target_cells)
    # EMD_1D(new_df, target_cells, ax[1])
    EMD1Dvs2D_Analysis (receptor_names, target_cells, signaling_receptor, new_df, ax[0], ax[1], ax[3])
    
    return f
