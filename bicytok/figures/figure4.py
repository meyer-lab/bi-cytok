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
from ..selectivityFuncs import getSampleAbundances, optimizeDesign, selecCalc, get_rec_vecs, minSelecFunc



path_here = dirname(dirname(__file__))

def makeFigure():  
    markerDF = importCITE()
    new_df = markerDF.head(1000)
    receptors = []
    for column in new_df.columns:
        if column not in ['CellType1', 'CellType2', 'CellType3', 'Cell']:
            receptors.append(column)
    ax, f = getSetup((40, 40), (1,1)) 
    cell_types = set()  # Using a set to store unique values

    for column in ['CellType1', 'CellType2', 'CellType3']:
        if column in new_df.columns:
            cell_types.update(new_df[column].tolist())
    cell_types = list(cell_types) 
    # parameters for selectivity 
    
    selectedDF1 = getSampleAbundances(receptors, cell_types, "CellType2") #figure out what column cd 8 t and treg are defined in 
    selectedDF2 = getSampleAbundances(receptors, cell_types, "CellType1") #figure out what column cd 8 t and treg are defined in 
    targCell1 = 'Treg'
    targCell2 = 'CD8 T'

    offTCells1 = [cell for cell in cell_types if cell != targCell1]
    offTCells2 = [cell for cell in cell_types if cell != targCell2]
    offTCells1 = np.array(offTCells1)
    offTCells2 = np.array(offTCells2)

    valency = 2
    prevOptAffs = [8.0, 8.0, 8.0]
    doseVec = np.logspace(-3, 3, num=20)
    secondary = 'CD122'
    resultsTreg = []
    resultsCD8T = []
    dose = .5

    #for _, dose in enumerate(doseVec):
    #for receptor in receptors:
    for receptor in receptors[:10]:
        selectivity1, _, _ = optimizeDesign(secondary, receptor, targCell1, offTCells1, selectedDF1, dose, valency, prevOptAffs)
        selectivity2, _, _ = optimizeDesign(secondary, receptor, targCell2, offTCells2, selectedDF2, dose, valency, prevOptAffs)
        resultsTreg.append({'Epitope': receptor, 'Dose': dose, 'Selectivity': selectivity1})
        resultsCD8T.append({'Epitope': receptor, 'Dose': dose, 'Selectivity': selectivity2})




    sorted_results_Treg = sorted(resultsTreg, key=lambda x: x['Selectivity'], reverse=True)[:10]
    top_epitopes_Treg = [result['Epitope'] for result in sorted_results_Treg]
    top_selectivity_values_Treg = [result['Selectivity'] for result in sorted_results_Treg]

    print('sorted_results_Treg:', sorted_results_Treg)
    # Fetch top 10 epitopes with highest selectivity values for CD8 T
    sorted_results_CD8T = sorted(resultsCD8T, key=lambda x: x['Selectivity'], reverse=True)[:10]
    top_epitopes_CD8T = [result['Epitope'] for result in sorted_results_CD8T]
    top_selectivity_values_CD8T = [result['Selectivity'] for result in sorted_results_CD8T]
    print('sorted_results_CD8T:', sorted_results_CD8T)

    bar_width = 0.4  # Width of the bars

    # Plotting the top 10 selectivity values for Treg on ax[0]
    ax[0].barh(np.arange(10), top_selectivity_values_Treg, height=bar_width, color='blue', alpha=0.6, label='Treg')

    # Plotting the top 10 selectivity values for CD8 T on ax[0] with an offset
    ax[0].barh(np.arange(10) + bar_width, top_selectivity_values_CD8T, height=bar_width, color='orange', alpha=0.6, label='CD8 T')

    # Adding labels for epitopes to the plot for Treg on ax[0]
    for i, (value, epitope) in enumerate(zip(top_selectivity_values_Treg, top_epitopes_Treg)):
        ax[0].text(value, i, f'{epitope}', ha='right', va='center', fontsize=15, color='blue')

    # Adding labels for epitopes to the plot for CD8 T on ax[0]
    for i, (value, epitope) in enumerate(zip(top_selectivity_values_CD8T, top_epitopes_CD8T)):
        ax[0].text(value, i + bar_width, f'{epitope}', ha='left', va='center', fontsize=15, color='red')

    # Adjusting plot aesthetics on ax[0]
    ax[0].set_yticks(np.arange(10) + bar_width / 2)
    ax[0].set_yticklabels(np.arange(1, 11), fontsize=15)
    ax[0].set_xlabel('Selectivity', fontsize=15)
    ax[0].set_ylabel('Epitope Rank', fontsize=15)
    ax[0].set_title('Top 10 Selectivity Values for Treg and CD8 T', fontsize=15)
    ax[0].legend(fontsize=15)


    '''
    target_cells = 'Treg'  
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
    
    # EMD_3D(new_df, target_cells, ax[0]) # just run this one line for 2D with Treg


 

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