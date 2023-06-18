from os.path import dirname, join
from .common import getSetup
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.optimize import minimize_scalar
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from ..BindingMod import polyc
from ..imports import importCITE
from scipy.optimize import minimize
import random

path_here = dirname(dirname(__file__))


def makeFigure(): 
    df = pd.DataFrame(importCITE())
    print (df)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

    # Plot IL2Ra distribution
    sns.boxplot(x='CellType1', y='CD25', data=df, ax=axes[0])
    axes[0].set_title('IL2Ra (CD25) Distribution')
    axes[0].set_xlabel('Cell Type')
    axes[0].set_ylabel('IL2Ra (CD25)')

    # Plot IL2RB distribution
    sns.boxplot(x='CellType1', y='CD122', data=df, ax=axes[1])
    axes[1].set_title('IL2RB (CD122) Distribution')
    axes[1].set_xlabel('Cell Type')
    axes[1].set_ylabel('IL2RB (CD122)')   
    
    def find_best_markers(cell_type, df):
        filtered_df = df[df['CellType1'] == cell_type]
        mean_expression = filtered_df.iloc[:, 2:].mean()
        mean_expression = mean_expression.drop(['CD25', 'CD122'])
        rest_of_cells = df[df['CellType1'] != cell_type]
        mean_expression_rest = rest_of_cells.iloc[:, 2:].mean()
        fold_change = mean_expression / mean_expression_rest
        sorted_markers = fold_change.sort_values(ascending=False)
        top_markers = sorted_markers.head(3)
        return top_markers.index.tolist()

    cell_type = 'NK'
    top_markers = find_best_markers(cell_type, df)
    print(f"The top markers for {cell_type} are: {top_markers}")


    
    # Randomly sample 1000 cells
    def optimize_ligand(marker, df):
        sample_df = df.sample(n=1000, random_state=51)
        sample_df = sample_df.reset_index(drop=True)

        for _, row in sample_df.iterrows():
            row['CD122'] = row['CD122'] * 1000
            row[marker] = row[marker] * 1000
                
       
        nk_marker = []
        nk_cd122 = []
        non_nk_marker = []
        non_nk_cd122 = []

        # Filter NK cells
        nk_df = sample_df[sample_df['CellType1'] == 'NK']
        nk_marker = nk_df[marker].tolist()
        nk_cd122 = nk_df['CD122'].tolist()

        # Filter non-NK cells
        non_nk_df = sample_df[sample_df['CellType1'] != 'NK']
        non_nk_marker = non_nk_df[marker].tolist()
        non_nk_cd122 = non_nk_df['CD122'].tolist()
        
        conc = 1e-9
        Kx = 1e-12
        Rtot_IL2NK = nk_cd122
        Rtot_IL2 = non_nk_cd122
        RtotmarkerNK = nk_marker
        Rtotmarker = non_nk_marker
        cplx_mono = np.array([[1]])
        Ctheta = np.array([1])
        KavIL2RB = np.array([[1e8]])
        Kavmarker_range = np.logspace(5, 10, num=100)
        
        # little queens store Rbound for the two markers on the two cell types 
        Rbound_NKMarker = []
        Rbound_marker = []
        Rbound_IL2NK = []
        Rbound_IL2 = []
        for Kavmarker in Kavmarker_range:
            for i, RtotmarkerNK in enumerate(RtotmarkerNK):
                _, Rbound_NKMarker_, _ = polyc(conc, Kx, np.array([RtotmarkerNK[i], Rtot_IL2[i]]), cplx_mono, Ctheta, np.array([Kavmarker]))
                Rbound_NKMarker.extend(Rbound_NKMarker_)
            
        
        

        for Rtot_IL2 in Rtot_IL2:
            _, Rbound_IL2_, _ = polyc(conc, Kx, np.array([Rtot_IL2]), cplx_mono, Ctheta, KavIL2RB)
            Rbound_IL2 .extend(Rbound_IL2_)
        ## MUST ADD HERE 
        litRatio = -np.sum(Rbound_IL2NK) / np.sum(Rbound_IL2)
        return litRatio
    
    result = minimize(optimize_ligand(top_markers[0], df), x0=1e8, bounds=[(1e5, 1e10)], method='L-BFGS-B')
    best_affinity = result.x[0]
    print("Best affinity value:", best_affinity)
    return fig