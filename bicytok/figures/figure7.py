#@Helen fix this later
from .common import getSetup
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from .common import EMD_2D
from ..imports import importCITE 
from .common import KL_divergence_2D
from .common import EMD_KL_clustermap


def makeFigure():  
    ''' clustermaps of either EMD or KL values for receptors + specified cell type'''
    markerDF = importCITE()
    new_df = markerDF.head(1000)
    receptors = []
    for column in new_df.columns:
        if column not in ['CellType1', 'CellType2', 'CellType3', 'Cell']:
            receptors.append(column)
    ax, f = getSetup((40, 40), (1,1)) 
    target_cells = "Treg"
   
    # Clustermap for EMD
    resultsEMD = []
    receptors = ['CD25', 'CD35']
    for receptor in receptors:
        val = EMD_2D(new_df, receptor, target_cells, ax = None) 
        resultsEMD.append(val)
    flattened_results = [result_tuple for inner_list in resultsEMD for result_tuple in inner_list]
    # Create a DataFrame from the flattened_results
    df_recep = pd.DataFrame(flattened_results, columns=['Distance', 'Receptor', 'Signal Receptor'])
    pivot_table = df_recep.pivot_table(index='Receptor', columns='Signal Receptor', values='Distance')
    f = EMD_KL_clustermap(pivot_table)

    
    # Clustermap for KL
    '''
    resultsKL = []
    for receptor in receptors[0:5]:
        val = KL_divergence_2D(new_df, receptor, target_cells, ax = None) 
        resultsKL.append(val)
        print('slay')
    flattened_resultsKL = [result_tuple for inner_list in resultsKL for result_tuple in inner_list]

    # Create a DataFrame from the flattened_results
    df_recep = pd.DataFrame(flattened_resultsKL, columns=['KLD', 'Receptor', 'Signal Receptor'])
    pivot_tableKL = df_recep.pivot_table(index='Receptor', columns='Signal Receptor', values='KLD')
    f = EMD_KL_clustermap(pivot_tableKL)
    '''
    
    return f     