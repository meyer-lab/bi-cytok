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


    def calculate_binding_ratio(df, marker):
        # Randomly sample 1000 cells
        sample_cells = df.sample(n=1000, replace=False)
        IL2RB_amounts = sample_cells['CD122'] * 1000
        CD335_amounts = sample_cells[marker] * 1000
        Kx = 1e-12
        Rtot_1 = np.array(IL2RB_amounts)
        cplx_mono = np.array([[1]])
        Ctheta = np.array([1])
        Kav = np.array([[1e8]])

        def objective_function(affinity_CD335):
            Kav[0, 0] = affinity_CD335
            _, Rbound, _ = polyc(1.0, Kx, Rtot_1, cplx_mono, Ctheta, Kav)
            NK_IL2RB_binding = np.sum(Rbound[:, 0])
            off_target_IL2RB_binding = np.sum(Rbound[:, 1:])
            ratio = NK_IL2RB_binding / off_target_IL2RB_binding
            return -ratio  # Negate the ratio since we want to maximize it

        # Perform optimization to determine the ligand's affinity for CD335
        bounds = [(1e5, 1e10)]  # Bounds for the affinity of the ligand for CD335
        initial_guess = 1e8  # Initial guess for the affinity of CD335
        result = minimize(objective_function, initial_guess, bounds=bounds)

        best_affinity_CD335 = result.x[0]
        best_ratio = -result.fun  # Retrieve the maximum ratio by negating the objective function value

        return best_ratio
    
    marker = 'CD335' 
    binding_ratio = calculate_binding_ratio(df, marker)
    print(f"The IL2RB binding ratio on NK cells compared to other cells for {marker} is: {binding_ratio}")
    return fig