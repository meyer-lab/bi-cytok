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
from .common import EMD_3D
from .common import EMD1Dvs2D_Analysis
from .common import Wass_KL_Dist
from ..imports import importCITE 
from .common import KL_divergence_2D
from .common import plot_kl_divergence_curves



path_here = dirname(dirname(__file__))

def makeFigure():  
    markerDF = importCITE()
    new_df = markerDF.head(1000)  
    new_df1 = markerDF.sample(n=10000, random_state=42) 
    new_df2 = markerDF.sample(n=10000, random_state=10)
    ax, f = getSetup((10, 10), (3, 2))
    target_cells = 'Treg'     
    signaling_receptor = 'CD122'     
    non_siganling_receptor = 'CD25'  
    receptor_names_top = ['CD25', 'Notch-2', 'CD4-1', 'CD27', 'CD278']  
    receptor_names_varried = ['CD25', 'CD109', 'CD27', 'TIGIT', 'CD28']
    # EMD_2D(new_df, signaling_receptor, target_cells, ax[0])
    # EMD_3D(new_df, signaling_receptor, target_cells, ax[2])
    # EMD_Distribution_Plot(ax[2], new_df2, signaling_receptor, non_siganling_receptor, target_cells)
    # EMD_1D(new_df, target_cells, ax[1])  
    # EMD1Dvs2D_Analysis (receptor_names_varried, target_cells, signaling_receptor, new_df, ax[0], ax[1], ax[2], ax[3])
    # KL_divergence_2D(new_df, 'CD122', "Treg", ax[0])
    plot_kl_divergence_curves(new_df, 'CD122', 'CD25', 'Treg', ax[1])
    plt.tight_layout()
    return f   