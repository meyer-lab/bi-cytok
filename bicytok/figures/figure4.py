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
    new_df = markerDF.head(100)
    receptors = []
    for column in new_df.columns:
        if column not in ['CellType1', 'CellType2', 'CellType3', 'Cell']:
            receptors.append(column)
    ax, f = getSetup((40, 40), (1,1))
    target_cells = 'Treg' 
    recep = 'CD122'
    # EMD_2D(new_df, recep, target_cells, ax[0])
    results = []
    for receptor in receptors:
        val = EMD_2D(new_df, receptor, target_cells, ax = None) # can make none
        results.append(val)

    receptor_names = [receptor for _, receptor in results[0]]

    # Create an empty matrix to store the EMD values
    emd_matrix = np.zeros((len(receptor_names), len(receptor_names)))

    # Fill in the matrix with EMD values from the results
    for i, receptor_x in enumerate(receptor_names):
        for j, receptor_y in enumerate(receptor_names):
            # Find the EMD value for the pair (receptor_x, receptor_y) in the results
            # If it's not found, you can set a default value or leave it as zero
            emd_value = 0.0  # Default value
            for result in results:
                if (receptor_x, receptor_y) in result:
                    emd_value = result[result.index((receptor_x, receptor_y))][1]
                    break
            emd_matrix[i, j] = emd_value


    # Create the heatmap on ax[0]
    ax[0].imshow(emd_matrix, cmap='viridis', interpolation='nearest')

    # Customize the heatmap appearance (e.g., add colorbar, labels)
    ax[0].set_xticks(range(len(receptor_names)))
    ax[0].set_xticklabels(receptor_names, rotation=90)
    ax[0].set_yticks(range(len(receptor_names)))
    ax[0].set_yticklabels(receptor_names)
    ax[0].set_xlabel('X-axis Receptor Names')
    ax[0].set_ylabel('Y-axis Receptor Names')
    ax[0].set_title('EMD Heatmap')

    return f     