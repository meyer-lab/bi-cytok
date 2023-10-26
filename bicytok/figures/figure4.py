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
from .common import EMD_3Dnew
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
    '''
    # EMD_2D(new_df, 'CD25', target_cells, ax = ax[0]) 
    # KL_divergence_2D(new_df, 'CD25', target_cells, ax[0])
    # receptors = ['CD25', 'CD8', 'CD122', 'CD35', 'CD314']
    
    target_cells_cd8t = 'CD8 T'  # CD8 T target cell type
    target_cells_treg = 'Treg'  # Treg target cell type
    
    resultsEMD_cd8t = []
    resultsEMD_treg = []
    
    for receptor in receptors:
        val_cd8t = EMD_2D(new_df, receptor, target_cells_cd8t, ax=None)
        val_treg = EMD_2D(new_df, receptor, target_cells_treg, ax=None)
        
        resultsEMD_cd8t.append(val_cd8t)
        resultsEMD_treg.append(val_treg)
        print('yas')
    
    flattened_results_cd8t = [result_tuple for inner_list in resultsEMD_cd8t for result_tuple in inner_list]
    flattened_results_treg = [result_tuple for inner_list in resultsEMD_treg for result_tuple in inner_list]
    
    df_recep_cd8t = pd.DataFrame(flattened_results_cd8t, columns=['Distance', 'Receptor', 'Signal Receptor'])
    df_recep_treg = pd.DataFrame(flattened_results_treg, columns=['Distance', 'Receptor', 'Signal Receptor'])
    
    df_recep_cd8t['Receptor Pair'] = df_recep_cd8t['Receptor'] + ' - ' + df_recep_cd8t['Signal Receptor']
    df_recep_treg['Receptor Pair'] = df_recep_treg['Receptor'] + ' - ' + df_recep_treg['Signal Receptor']

    # Sort the DataFrames by 'Distance' in descending order
    df_sorted_cd8t = df_recep_cd8t.sort_values(by='Distance', ascending=False)
    df_sorted_treg = df_recep_treg.sort_values(by='Distance', ascending=False)

    # Select the top 10 rows
    top_10_cd8t = df_sorted_cd8t.head(10)
    top_10_treg = df_sorted_treg.head(10)

    distance_values_cd8t = top_10_cd8t['Distance']
    distance_values_treg = top_10_treg['Distance']
    
    receptor_pairs_cd8t = top_10_cd8t['Receptor Pair']
    receptor_pairs_treg = top_10_treg['Receptor Pair']

    # Create a combined bar plot on ax[0] with pink and light blue bars
    width = 0.35  # Adjust the width of the bars
    ax[0].bar(receptor_pairs_cd8t, distance_values_cd8t, width, color='pink', label='CD8 T')  # Bars for CD8 T
    ax[0].bar(receptor_pairs_treg, distance_values_treg, width, color='lightblue', label='Treg', align='edge')  # Bars for Treg

    # Add labels and a title to the subplot
    ax[0].set_ylabel('Distance', fontsize=30)
    ax[0].set_xlabel('Receptor Pairs', fontsize=20)

    # Add the title with a larger font size
    title = ax[0].set_title(f'Top 10 Receptor Pairs by Distance')
    title.set_fontsize(45)  # Adjust the fontsize as needed

    ax[0].set_xticks(np.arange(len(receptor_pairs_cd8t) + len(receptor_pairs_treg)))
    ax[0].set_xticklabels(receptor_pairs_cd8t.tolist() + receptor_pairs_treg.tolist(), rotation=0)
    
    # Add a legend to differentiate CD8 T and Treg bars
    ax[0].legend(loc='upper right', fontsize=20)
    '''
    
    print(EMD_3D(new_df, target_cells, ax[0])) # just run this one line for 2D with Treg




    '''
    resultsEMD2D = [] 
    for receptor in receptors:
        val = EMD_2D(new_df, receptor, target_cells, ax = None) 
        resultsKL.append(val)
        print('slay')
    flattened_resultsEMD2D = [result_tuple for inner_list in resultsEMD2D for result_tuple in inner_list]
    df_recep = pd.DataFrame(flattened_resultsEMD2D, columns=['Distance', 'Receptor', 'Signal Receptor'])
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
    val = EMD_3D(new_df, target_cells, ax = None) # just run this one line for 2D with Treg
    resultsEMD3D.append(val)
    flattened_resultsEMD3D = [result_tuple for inner_list in resultsEMD3D for result_tuple in inner_list]
    df_recep = pd.DataFrame(flattened_resultsEMD3D, columns=['Distance', 'Receptor', 'Signal Receptor'])
    pivot_table3D = df_recep.pivot_table(index='Receptor', columns='Signal Receptor', values='Distance')
    f = EMD3D_clustermap(pivot_table3D)
    
    # f = EMD_clustermap(pivot_table) 
    # f = KLD_clustermap(pivot_tableKL) 
    
    '''
    return f     