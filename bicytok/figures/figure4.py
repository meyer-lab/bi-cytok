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
from .common import Wass_KL_Dist
from ..imports import importCITE 



path_here = dirname(dirname(__file__))

def makeFigure():  
    markerDF = importCITE()
    new_df = markerDF.head(1000) 
    ax, f = getSetup((10, 10), (3, 2))
    target_cells = 'Treg' 
    signaling_receptor = 'CD122' 
    non_siganling_receptor = 'CD25'
    receptor_names_neartop = ['CD4-2', 'CD25', 'CD267', 'CD4-1', 'CD272'] 
    receptor_names_top = ['CD4-2', 'CD25', 'CD140a', 'CD267', 'CD252']
    receptor_names_varried = ['CD4-2', 'CD25', 'CD4-1', 'CD59', 'CD26-2']
    # EMD_2D(new_df, signaling_receptor, target_cells, ax[0])
    # EMD_Distribution_Plot(ax[2], new_df, signaling_receptor, non_siganling_receptor, target_cells)
    EMD_1D(new_df, target_cells, ax[3])
    # EMD1Dvs2D_Analysis (receptor_names_varried, target_cells, signaling_receptor, new_df, ax[0], ax[1], ax[2], ax[3])
       
    return f 
