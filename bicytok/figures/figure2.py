"""
This creates Figure 2, cell type ratios of response of bispecific IL-2 cytokines at varin abundances using binding model.
"""

import numpy as np
import pandas as pd
import seaborn as sns
from .common import getSetup
from ..MBmodel import runFullModel_bispec


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    conc = np.array([1])
    modelDF = runFullModel_bispec(conc)

    # print(modelDF)

    cells = ["Treg", "Thelper", "NK", "CD8"]
    ax, f = getSetup((22, 30), (3, 2))

    signalRatio(ax[0], modelDF, "Treg", "NK", "Low")
    signalRatio(ax[1], modelDF, "Treg", "CD8", "Low")
    signalRatio(ax[2], modelDF, "Treg", "NK", "Medium")
    signalRatio(ax[3], modelDF, "Treg", "CD8", "Medium")
    signalRatio(ax[4], modelDF, "Treg", "NK", "High")
    signalRatio(ax[5], modelDF, "Treg", "CD8", "High")

    return f

# calc at different valencies


def signalRatio(ax, dataframe, cellType1, cellType2, affinity):
    ratios_DF = pd.DataFrame(columns={"X-Abundance", "Y-Abundance", "Ratio"})

    type1_vals = dataframe.loc[(dataframe.Cell == cellType1) & (dataframe.Affinity == affinity)]
    type2_vals = dataframe.loc[(dataframe.Cell == cellType2) & (dataframe.Affinity == affinity)]

    abundances = dataframe.loc[(dataframe.Cell == cellType1) & (dataframe.Affinity == 'Medium')]["Abundance"]

    for x_abundance in abundances:
        x_signal = type1_vals.loc[(type1_vals.Abundance == x_abundance)]["Predicted"]
        for y_abundance in abundances:
            y_signal = type2_vals.loc[(type2_vals.Abundance == y_abundance)]["Predicted"]
            signalRatio = x_signal / y_signal
            ratios_DF = ratios_DF.append(pd.DataFrame({"X-Abundance": x_abundance, "Y-Abundance": y_abundance, "Ratio": signalRatio}))

    result = ratios_DF.pivot(index="Y-Abundance", columns="X-Abundance", values="Ratio")

    result = result[::-1]

    title = cellType1 + "/" + cellType2 + " at " + affinity + " Affinity"
    xlabel = "Epitope Abundance on " + cellType1
    ylabel = "Epitope Abundance on " + cellType2

    sns.heatmap(result, cmap='RdYlGn', ax=ax, cbar_kws={'label': 'Ratio of Cell Signal'})
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)

    return ratios_DF
