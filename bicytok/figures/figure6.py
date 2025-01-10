from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.neighbors import KernelDensity

from ..imports import importCITE
from .common import getSetup

path_here = Path(__file__).parent.parent

plt.rcParams["svg.fonttype"] = "none"


def makeFigure():
    """
    Figure file to generate bar plots of 1D KL divergence and EMD values of
    most unique receptor for each given cell type/subset.
    toRemove: cell types to remove from calculations and figure generation
    cellLevel: cell type categorization level, see cell types/subsets in CITE data
    """

    ax, f = getSetup((12, 4), (1, 2))

    CITE_DF = importCITE()
    cellLevel = "CellType2"

    cells = list(CITE_DF[cellLevel].unique())
    toRemove = [
        "Eryth",
        "MAIT",
        "ASDC",
        "Platelet",
        "gdT",
        "dnT",
        "B intermediate",
        "CD4 CTL",
        "NK Proliferating",
        "CD8 Proliferating",
        "CD4 Proliferating",
    ]
    for removed in toRemove:
        cells.remove(removed)

    df = pd.DataFrame(columns=["Cell", "KL Divergence", "Earth Mover's Distance"])

    for targCell in cells:
        markerDF = pd.DataFrame(columns=["Marker", "KL", "EMD"])

        non_marker_columns = ["CellType1", "CellType2", "CellType3", "Cell"]
        marker_columns = CITE_DF.columns[~CITE_DF.columns.isin(non_marker_columns)]
        markerDF = CITE_DF.loc[:, marker_columns]

        for marker in CITE_DF.columns:
            markAvg = np.mean(CITE_DF[marker].values)
            if markAvg > 0.0001:
                targCellMark = (
                    CITE_DF.loc[CITE_DF[cellLevel] == targCell][marker].values / markAvg
                )
                offTargCellMark = (
                    CITE_DF.loc[CITE_DF[cellLevel] != targCell][marker].values / markAvg
                )
                if np.mean(targCellMark) > np.mean(offTargCellMark):
                    kdeTarg = KernelDensity(kernel="gaussian").fit(
                        targCellMark.reshape(-1, 1)
                    )
                    kdeOffTarg = KernelDensity(kernel="gaussian").fit(
                        offTargCellMark.reshape(-1, 1)
                    )
                    minVal = np.minimum(targCellMark.min(), offTargCellMark.min()) - 10
                    maxVal = np.maximum(targCellMark.max(), offTargCellMark.max()) + 10
                    outcomes = np.arange(minVal, maxVal + 1).reshape(-1, 1)
                    distTarg = np.exp(kdeTarg.score_samples(outcomes))
                    distOffTarg = np.exp(kdeOffTarg.score_samples(outcomes))
                    KL_div = stats.entropy(
                        distOffTarg.flatten() + 1e-200,
                        distTarg.flatten() + 1e-200,
                        base=2,
                    )
                    markerDF = pd.concat(
                        [
                            markerDF,
                            pd.DataFrame(
                                {
                                    "Marker": [marker],
                                    "KL": KL_div,
                                    "EMD": stats.wasserstein_distance(
                                        targCellMark, offTargCellMark
                                    ),
                                }
                            ),
                        ],
                        ignore_index=True,
                    )
        KLrow = markerDF.iloc[markerDF["KL"].idxmax()]
        markerKL = KLrow["Marker"]
        KL = KLrow["KL"]
        EMDrow = markerDF.iloc[markerDF["EMD"].idxmax()]
        markerEMD = EMDrow["Marker"]
        EMD = EMDrow["EMD"]

        data = {
            "Cell": targCell,
            "KL Marker": targCell + " " + markerKL,
            "KL Divergence": [KL],
            "EMD Marker": targCell + " " + markerEMD,
            "Earth Mover's Distance": [EMD],
        }
        df_temp = pd.DataFrame(
            data,
            columns=[
                "Cell",
                "KL Marker",
                "KL Divergence",
                "EMD Marker",
                "Earth Mover's Distance",
            ],
        )
        df = pd.concat([df, df_temp], ignore_index=True)

    sns.barplot(
        data=df.sort_values(by=["KL Divergence"]),
        x="KL Marker",
        y="KL Divergence",
        ax=ax[0],
    )
    sns.barplot(
        data=df.sort_values(by=["Earth Mover's Distance"]),
        x="EMD Marker",
        y="Earth Mover's Distance",
        ax=ax[1],
    )
    ax[0].set_xticklabels(
        labels=ax[0].get_xticklabels(), rotation=45, horizontalalignment="right"
    )
    ax[1].set_xticklabels(
        labels=ax[1].get_xticklabels(), rotation=45, horizontalalignment="right"
    )

    return f
