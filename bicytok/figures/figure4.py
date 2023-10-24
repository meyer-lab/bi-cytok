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
from .common import EMD3D_clustermap
from .common import EMD1Dvs2D_Analysis
from .common import Wass_KL_Dist
from ..imports import importCITE 
from .common import plot_kl_divergence_curves
from .common import plot_2d_density_visualization
from .common import EMD_2D_pair
from .common import calculate_kl_divergence_2D
from .common import KL_divergence_2D
from .common import KLD_clustermap
from .common import EMD_clustermap



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
    # EMD_2D(new_df, 'CD25', target_cells, ax = ax[0]) 
    # KL_divergence_2D(new_df, 'CD25', target_cells, ax[0])
    receptors = ['CD25']
    
    resultsEMD = []
    for receptor in receptors:
        val = EMD_2D(new_df, receptor, target_cells, ax = None) 
        resultsEMD.append(val)
        print ('yas')
    flattened_results = [result_tuple for inner_list in resultsEMD for result_tuple in inner_list]
    
    df_recep = pd.DataFrame(flattened_results, columns=['Distance', 'Receptor', 'Signal Receptor'])
    df_recep['Receptor Pair'] = df_recep['Receptor'] + ' - ' + df_recep['Signal Receptor']

    # Sort the DataFrame by 'Distance' in descending order
    df_sorted = df_recep.sort_values(by='Distance', ascending=False)

    # Select the top 10 rows
    top_10 = df_sorted.head(10)

    distance_values = top_10['Distance']
    receptor_pairs = top_10['Receptor Pair']

    # Create a bar plot on ax[0]
    ax[0].bar(receptor_pairs, distance_values, color='skyblue')  # Horizontal bar plot

    # Add labels and a title to the subplot
    ax[0].set_ylabel('Distance', fontsize=30)
    ax[0].set_xlabel('Receptor Pairs', fontsize=20)

    # Add the title with a larger font size
    title = ax[0].set_title(f'Top 10 Receptor Pairs for {target_cells} by Distance')
    title.set_fontsize(45)  # Adjust the fontsize as needed

    # Optionally, rotate the labels on the y-axis for better readability
    ax[0].set_xticklabels(receptor_pairs, rotation=0)
    # Adjust the fontsize of the tick labels
    ax[0].tick_params(axis='both', labelsize=20) 
    '''
    pivot_table = df_recep.pivot_table(index='Receptor', columns='Signal Receptor', values='Distance')
    '''
    '''
    resultsKL = [] 
    for receptor in receptors:
        val = KL_divergence_2D(new_df, receptor, target_cells, ax = None) 
        resultsKL.append(val)
        print('slay')
    flattened_resultsKL = [result_tuple for inner_list in resultsKL for result_tuple in inner_list]
    df_recep = pd.DataFrame(flattened_resultsKL, columns=['KLD', 'Receptor', 'Signal Receptor'])
    pivot_tableKL = df_recep.pivot_table(index='Receptor', columns='Signal Receptor', values='KLD')
    '''
    '''
    resultsEMD3D = []
    val = EMD_3D(new_df, target_cells, ax = None) 
    resultsEMD3D.append(val)
    flattened_results = [result_tuple for inner_list in resultsEMD3D for result_tuple in inner_list]
    df_recep = pd.DataFrame(flattened_results, columns=['Distance', 'Receptor', 'Signal Receptor'])
    pivot_table3D = df_recep.pivot_table(index='Receptor', columns='Signal Receptor', values='Distance')
    # f = EMD_clustermap(pivot_table) 
    
    # f = KLD_clustermap(pivot_tableKL) 
    f = EMD3D_clustermap(pivot_table3D)
    '''
    return f     