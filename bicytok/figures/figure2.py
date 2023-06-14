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

    def optimize_binding_affinity(cell_type, marker_name):
            # Step A: Grab a random sample of 1000 cells from the CITE-seq dataset
        df_sample = df.sample(n=1000, random_state=42)  # Adjust the random_state as desired

        # Step B: Calculate IL2RB amount and marker amount for each cell individually
        il2rb_amounts = df_sample[df_sample['CellType1'] == cell_type]['CD122'] * 1000
        marker_amounts = df_sample[df_sample['CellType1'] == cell_type][marker_name] * 1000

        # Step C: Use optimization to determine the best affinity
        best_affinity = None
        best_ratio = -float('inf')

        def binding_residuals(affinity):
            # Run binding model 1000 times and calculate IL2RB bound ratios
            il2rb_bound = []
            marker_bound = []
            for _ in range(1000):
                il2rb_bound_val, marker_bound_val, _ = polyc(1, 1e-12, il2rb_amounts, [1], [1], np.array([1e8, affinity]))
                il2rb_bound.append(il2rb_bound_val[0])
                marker_bound.append(marker_bound_val[0])

            # Calculate IL2RB bound ratio
            ratio = sum(il2rb_bound) / sum(marker_bound)
            return 1 - ratio  # Minimize the residuals

        # Perform optimization with bounds on affinity
        result = minimize_scalar(binding_residuals, bounds=(1e5, 1e10), method='bounded')
        best_affinity = result.x
        best_ratio = 1 - result.fun

        return best_affinity, best_ratio
    
    return fig