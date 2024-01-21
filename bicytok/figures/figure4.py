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
    ax, f = getSetup((10, 10), (1,1)) 
    
    target_cells = 'Treg' 

    resultsEMD2D = [] 
    for receptor in receptors:
        val = EMD_2D(new_df, receptor, target_cells, ax = None)  
        resultsEMD2D.append(val)
        print ('slay')

    flattened_resultsEMD2D = [result_tuple for inner_list in resultsEMD2D for result_tuple in inner_list]
    df_recep = pd.DataFrame(flattened_resultsEMD2D, columns=['Distance', 'Receptor', 'Signal Receptor'])
    df_recep_top100 = df_recep.nlargest(100, 'Distance')

    pivot_table = df_recep_top100.pivot_table(index='Receptor', columns='Signal Receptor', values='Distance')

    
    
   
    
    f = EMD_clustermap(pivot_table)
    f.ax_heatmap.set_xticklabels(f.ax_heatmap.get_xticklabels(), fontsize=12)
    f.ax_heatmap.set_yticklabels(f.ax_heatmap.get_yticklabels(), fontsize=12)
    f.fig.suptitle("Top EMD 2D values for Treg Cells", fontsize=16)
    
    
    '''
    resultsKL = [] 
    for receptor in receptors:
        val = KL_divergence_2D(new_df, receptor, target_cells, ax = None) 
        resultsKL.append(val)
    flattened_resultsKL = [result_tuple for inner_list in resultsKL for result_tuple in inner_list]
    df_recep = pd.DataFrame(flattened_resultsKL, columns=['KLD', 'Receptor', 'Signal Receptor'])
    pivot_tableKL = df_recep.pivot_table(index='Receptor', columns='Signal Receptor', values='KLD')
    f = KLD_clustermap(pivot_tableKL) 
    f.ax_heatmap.set_xticklabels(f.ax_heatmap.get_xticklabels(), fontsize=12)
    f.ax_heatmap.set_yticklabels(f.ax_heatmap.get_yticklabels(), fontsize=12)
    f.fig.suptitle("Top 2D KL values for Treg Cells", fontsize=16)
    '''
    return f    