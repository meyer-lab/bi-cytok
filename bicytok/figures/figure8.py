import pandas as pd
import seaborn as sns

from ..distanceMetricFuncs import KL_divergence_2D
from ..imports import importCITE
from .common import getSetup


TARGET_CELL = "Treg"


def makeFigure():
    """clustermaps of KL values for receptors + specified cell type"""
    ax, f = getSetup((40, 40), (1, 1))

    markerDF = importCITE()
    new_df = markerDF.head(1000)
    receptors = []
    for column in new_df.columns:
        if column not in ["CellType1", "CellType2", "CellType3", "Cell"]:
            receptors.append(column)

    receptors = ["CD25", "CD35"]
    TARGET_CELL = "Treg"
    resultsKL = []
    for receptor in receptors[0:5]:
        val = KL_divergence_2D(
            new_df, receptor, TARGET_CELL, special_receptor=None, ax=None
        )
        resultsKL.append(val)
    flattened_resultsKL = [
        result_tuple for inner_list in resultsKL for result_tuple in inner_list
    ]

    # Create a DataFrame from the flattened_results
    df_recep = pd.DataFrame(
        flattened_resultsKL, columns=["KLD", "Receptor", "Signal Receptor"]
    )
    pivot_tableKL = df_recep.pivot_table(
        index="Receptor", columns="Signal Receptor", values="KLD"
    )
    dataset = pivot_tableKL.fillna(0)
    f = sns.clustermap(
        dataset, cmap="bwr", figsize=(10, 10), annot_kws={"fontsize": 16}
    )

    return f
