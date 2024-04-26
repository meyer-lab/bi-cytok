from .common import getSetup
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from ..distanceMetricFuncs import EMD_2D, KL_divergence_2D
from ..imports import importCITE


def makeFigure():
    """clustermaps of either EMD values for receptors + specified cell type"""
    markerDF = importCITE()
    new_df = markerDF.head(1000)
    receptors = []
    for column in new_df.columns:
        if column not in ["CellType1", "CellType2", "CellType3", "Cell"]:
            receptors.append(column)
    ax, f = getSetup((40, 40), (1, 1))
    target_cells = "Treg"

    # Clustermap for EMD
    resultsEMD = []
    receptors = ["CD25", "CD35"]
    for receptor in receptors:
        val = EMD_2D(new_df, receptor, target_cells, special_receptor=None, ax=None)
        resultsEMD.append(val)
    flattened_results = [
        result_tuple for inner_list in resultsEMD for result_tuple in inner_list
    ]
    # Create a DataFrame from the flattened_results
    df_recep = pd.DataFrame(
        flattened_results, columns=["Distance", "Receptor", "Signal Receptor"]
    )
    pivot_table = df_recep.pivot_table(
        index="Receptor", columns="Signal Receptor", values="Distance"
    )
    dataset = pivot_table.fillna(0)
    f = sns.clustermap(
        dataset, cmap="bwr", figsize=(10, 10), annot_kws={"fontsize": 16}
    )

    return f
