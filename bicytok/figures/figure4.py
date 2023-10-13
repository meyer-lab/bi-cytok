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
from .common import calculate_kl_divergence_2D
from .common import KL_divergence_2D



path_here = dirname(dirname(__file__))

def makeFigure():  
    markerDF = importCITE()
    new_df = markerDF.head(1000)
    receptors = []
    for column in new_df.columns:
        if column not in ['CellType1', 'CellType2', 'CellType3', 'Cell']:
            receptors.append(column)
    ax, f = getSetup((40, 40), (1,1))
    target_cells = 'Treg' 
    ######################################################
    '''
    resultsEMD = []
    for receptor in receptors: 
        val = EMD_2D(new_df, receptor, target_cells, ax = None) 
        print(val)
        resultsEMD.append(val)
    flattened_results = [result_tuple for inner_list in resultsEMD for result_tuple in inner_list]
    # Create a DataFrame from the flattened_results
    df_recep = pd.DataFrame(flattened_results, columns=['Distance', 'Receptor', 'Signal Receptor'])
    pivot_table = df_recep.pivot_table(index='Receptor', columns='Signal Receptor', values='Distance')
    # Create the heatmap on ax[0] 
    sns.heatmap(pivot_table, annot=False, fmt='.2f', cmap='viridis', ax=ax[0])

    # Customize the heatmap appearance (e.g., add colorbar, labels)
    ax[0].set_xlabel('Receptor')
    ax[0].set_ylabel('Receptor')
    ax[0].set_title('EMD Heatmap')
    ######################################################
    '''
    
    resultsKL = []
    for receptor in receptors[0:5]:
        val = KL_divergence_2D(new_df, receptor, target_cells, ax = None) 
        resultsKL.append(val)
    flattened_resultsKL = [result_tuple for inner_list in resultsKL for result_tuple in inner_list]

    # Create a DataFrame from the flattened_results
    df_recep = pd.DataFrame(flattened_resultsKL, columns=['KLD', 'Receptor', 'Signal Receptor'])
    pivot_tableKL = df_recep.pivot_table(index='Receptor', columns='Signal Receptor', values='KLD')
    # Create the heatmap on ax[0] 
    sns.heatmap(pivot_tableKL, annot=False, fmt='.2f', cmap='viridis', ax=ax[0])

    # Customize the heatmap appearance (e.g., add colorbar, labels)
    ax[0].set_xlabel('Receptor')
    ax[0].set_ylabel('Receptor')
    ax[0].set_title('KL Heatmap')

    return f     