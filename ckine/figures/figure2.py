"""
This creates Figure 2, cell type ratios of response of bispecific IL-2 cytokines at varin abundances using binding model.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.linalg.decomp_svd import null_space
from .figureCommon import getSetup, plotBispecific
from ..MBmodel import runFullModel_bispec


def makeFigure():
    """Get a list of the axis objects and create a figure"""

    modelDF = runFullModel_bispec(time=[0.5, 1])

    # print(modelDF)

    #treg_nk_rat = signalRatio(modelDF, "Treg","NK")

    cells = ["Treg", "Thelper", "NK", "CD8"]
    ax, f = getSetup((20, 10), (2, 1))

    signalRatio(ax[0], modelDF, "Treg", "NK")
    signalRatio(ax[1], modelDF, "Treg", "CD8")

    return f

# calc at different valencies


def signalRatio(ax, dataframe, cellType1, cellType2):
    ratios_DF = pd.DataFrame(columns={"X-Abundance", "Y-Abundance", "Ratio"})

    type1_vals = dataframe.loc[(dataframe.Cell == cellType1) & (dataframe.Affinity == 'Medium')]
    type2_vals = dataframe.loc[(dataframe.Cell == cellType2) & (dataframe.Affinity == 'Medium')]

    abundances = dataframe.loc[(dataframe.Cell == cellType1) & (dataframe.Affinity == 'Medium')]["Abundance"]

    for x_abundance in abundances:
        x_signal = type1_vals.loc[(type1_vals.Abundance == x_abundance)]["Predicted"]
        for y_abundance in abundances:
            y_signal = type2_vals.loc[(type2_vals.Abundance == y_abundance)]["Predicted"]
            signalRatio = x_signal / y_signal
            ratios_DF = ratios_DF.append(pd.DataFrame({"X-Abundance": x_abundance, "Y-Abundance": y_abundance, "Ratio": signalRatio}))

    test = (np.asarray(ratios_DF["Ratio"])).reshape(34, 34)
    result = ratios_DF.pivot(index="Y-Abundance", columns="X-Abundance", values="Ratio")

    title = cellType1 + "/" + cellType2
    xlabel = "Epitope Abundance on " + cellType1
    ylabel = "Epitope Abundance on " + cellType2

    sns.heatmap(result, cmap='RdYlGn', ax=ax)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)

    return ratios_DF
