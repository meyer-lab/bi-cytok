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
from .common import KL_divergence_forheatmap
from .common import plot_kl_divergence_curves
from .common import plot_2d_density_visualization
from .common import EMD_2D_pair
from .common import KL_divergence_2D_pair
from .common import bindingmodel_selectivity_pair



path_here = dirname(dirname(__file__))

def makeFigure():  
    markerDF = importCITE()
    new_df = markerDF.head(5)
    receptors = []
    for column in new_df.columns:
        if column not in ['CellType1', 'CellType2', 'CellType3', 'Cell']:
            receptors.append(column)
    ax, f = getSetup((40, 40), (1,1))
    target_cells = 'Treg' 

    results_matrix = []    
    for receptor in receptors:
        current_result = KL_divergence_forheatmap(new_df, receptor, target_cells)
        results_matrix.append(current_result)

    ax[0].set_title('KL Divergence Heatmap')
    ax[0].set_xticks(np.arange(len(receptors)))
    ax[0].set_yticks(np.arange(len(receptors)))
    ax[0].set_xticklabels(receptors)
    ax[0].set_yticklabels(receptors)
    sns.heatmap(results_matrix, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax[0])
 
 
    return f     