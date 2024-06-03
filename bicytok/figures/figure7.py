import pandas as pd
import seaborn as sns

from ..distanceMetricFuncs import EMD_2D
from ..imports import importCITE
from .common import getSetup


TARGET_CELL = "Treg"


def makeFigure():
    """Figure to generate clustermaps of EMD values for receptors
    + specified cell type"""
    ax, f = getSetup((40, 40), (1, 1))

    markerDF = importCITE()
    new_df = markerDF.head(1000)
    receptors = []
    for column in new_df.columns:
        if column not in ["CellType1", "CellType2", "CellType3", "Cell"]:
            receptors.append(column)

    # Clustermap for EMD
    resultsEMD = []
    receptors = ["CD25", "CD35"]
    for receptor in receptors:
        val = EMD_2D(new_df, receptor, TARGET_CELL, special_receptor=None, ax=None)
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
