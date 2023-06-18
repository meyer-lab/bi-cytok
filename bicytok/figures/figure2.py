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


    
    sample_df = df.sample(n=1000, random_state=51)
    sample_df = sample_df.reset_index(drop=True)

    for _, row in sample_df.iterrows():
        row['CD122'] = row['CD122'] * 1000
        row['CD335'] = row['CD335'] * 1000
                
       
    nk_marker = []
    nk_cd122 = []
    non_nk_marker = []
    non_nk_cd122 = []

    # Filter NK cells
    nk_df = sample_df[sample_df['CellType1'] == 'NK']
    nk_marker = nk_df['CD335'].tolist()
    nk_cd122 = nk_df['CD122'].tolist()

    # Filter non-NK cells
    non_nk_df = sample_df[sample_df['CellType1'] != 'NK']
    non_nk_marker = non_nk_df['CD335'].tolist()
    non_nk_cd122 = non_nk_df['CD122'].tolist()
        
    def optimization_function(Kav): 
        Kav_scalar = Kav[0]
        Kav_matrix = np.array([[Kav_scalar, 1e2], [1e2, 1e8]])
        affinity_values = []
        
        for i in range(len(nk_marker)):
            L0 = 1e-9
            KxStar = 1e-12
            cplx = np.array([[1, 0], [0, 1]])
            Ctheta = np.array([1, 1])
            Rtot_IL2 = non_nk_cd122[i]
            RtotmarkerNK = nk_marker[i]
            Rtot_array = np.array([RtotmarkerNK, Rtot_IL2])

            _, Rbound_NKMarker_, _ = polyc(L0, KxStar, Rtot_array, cplx, Ctheta, Kav_matrix)
            
            ratio = np.sum(Rbound_NKMarker_[:, 0]) / np.sum(Rbound_NKMarker_[:, 1])
        
            affinity_values.append(ratio)
        return -np.sum(affinity_values)
    
    initial_guess = [1e5]
    result = minimize(optimization_function, initial_guess, bounds=[(1e5, 1e10)])
    optimized_Kav = result.x[0]
    print(f"Optimized Kav: {optimized_Kav}")

    return fig