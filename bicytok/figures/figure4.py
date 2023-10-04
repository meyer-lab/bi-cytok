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

    
    cd8_t_df = new_df[new_df['CellType2'] == 'Treg']
    off_target_df = new_df[new_df['CellType2'] != 'Treg'] 
    
    receptor_columns = ['CD25', 'CD57']
    receptor_df = cd8_t_df[receptor_columns]
    receptor_df_off_target = off_target_df[receptor_columns]
    print(cd8_t_df['CD57'].mean())
    print(receptor_df_off_target['CD57'].mean())
    ######################################################
    sns.kdeplot(cd8_t_df['CD25'], ax=ax[0], label='CD25 (CD8 T)', shade=True)
    sns.kdeplot(cd8_t_df['CD57'], ax=ax[0], label='CD57 (CD8 T)', shade=True)
    sns.kdeplot(receptor_df_off_target['CD25'], ax=ax[0], label='CD25 (Off-Target)', shade=True)
    sns.kdeplot(receptor_df_off_target['CD57'], ax=ax[0], label='CD57 (Off-Target)', shade=True)
    ax[0].set_title('Receptor Distributions for CD8 T Cells and Off-Target Cells')    
    ax[0].set_xlabel('Receptor Expression')
    #ax[0].set_xlim(0, 600)
    ax[0].legend()
    
    '''
    resultsEMD = []

    for receptor in receptors:
        val = EMD_2D(new_df, receptor, target_cells, ax = None) 
        resultsEMD.append(val)
        print ('working')
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
    
    resultsKL = []
    for receptor in receptors:
        val = KL_divergence_2D(new_df, receptor, target_cells, ax = None) 
        resultsKL.append(val)
        print('slay')
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
    # f = KLD_clustermap(pivot_tableKL)
    
    # f = EMD_clustermap(pivot_table)
    
   
    f = KLD_clustermap(pivot_tableKL)
    # you are running cluster EMD look for title slay 
    # f = EMD_clustermap(pivot_table)
    '''

    return f     