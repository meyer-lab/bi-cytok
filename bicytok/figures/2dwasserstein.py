from os.path import dirname, join
from .common import getSetup
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from ..selectivityFuncs import get_cell_bindings, getSampleAbundances, get_rec_vecs, optimizeDesign, minSelecFunc
from ..imports import importCITE
from pandas.api.types import CategoricalDtype
from sklearn.neighbors import KernelDensity
from scipy import stats




path_here = dirname(dirname(__file__))

def makeFigure():
    ax, f = getSetup((10, 8), (2, 2)) 

    def Wass_KL_Dist(ax, targCell, numFactors, RNA=False, offTargState=0):
        """Finds markers which have average greatest difference from other cells"""
        CITE_DF = importCITE()

        markerDF = pd.DataFrame(columns=["Marker", "Cell Type", "Amount"])
        for marker in CITE_DF.loc[:, ((CITE_DF.columns != 'CellType1') & (CITE_DF.columns != 'CellType2') & (CITE_DF.columns != 'CellType3') & (CITE_DF.columns != 'Cell'))].columns:
            markAvg = np.mean(CITE_DF[marker].values)
            if markAvg > 0.0001:
                targCellMark = CITE_DF.loc[CITE_DF["CellType3"] == targCell][marker].values / markAvg
                # Compare to all non-memory Tregs
                if offTargState == 0:
                    offTargCellMark = CITE_DF.loc[CITE_DF["CellType3"] != targCell][marker].values / markAvg
                # Compare to all non-Tregs
                elif offTargState == 1:
                    offTargCellMark = CITE_DF.loc[CITE_DF["CellType2"] != "Treg"][marker].values / markAvg
                # Compare to naive Tregs
                elif offTargState == 2:
                    offTargCellMark = CITE_DF.loc[CITE_DF["CellType3"] == "Treg Naive"][marker].values / markAvg
                if np.mean(targCellMark) > np.mean(offTargCellMark):
                    kdeTarg = KernelDensity(kernel='gaussian').fit(targCellMark.reshape(-1, 1))
                    kdeOffTarg = KernelDensity(kernel='gaussian').fit(offTargCellMark.reshape(-1, 1))
                    minVal = np.minimum(targCellMark.min(), offTargCellMark.min()) - 10
                    maxVal = np.maximum(targCellMark.max(), offTargCellMark.max()) + 10
                    outcomes = np.arange(minVal, maxVal + 1).reshape(-1, 1)
                    distTarg = np.exp(kdeTarg.score_samples(outcomes))
                    distOffTarg = np.exp(kdeOffTarg.score_samples(outcomes))
                    KL_div = stats.entropy(distOffTarg.flatten() + 1e-200, distTarg.flatten() + 1e-200, base=2)
                    markerDF = pd.concat([markerDF, pd.DataFrame({"Marker": [marker], "Wasserstein Distance": stats.wasserstein_distance(targCellMark, offTargCellMark), "KL Divergence": KL_div})])

        corrsDF = pd.DataFrame()
        for i, distance in enumerate(["Wasserstein Distance", "KL Divergence"]):
            ratioDF = markerDF.sort_values(by=distance)
            posCorrs = ratioDF.tail(numFactors).Marker.values
            corrsDF = pd.concat([corrsDF, pd.DataFrame({"Distance": distance, "Marker": posCorrs})])
            markerDF = markerDF.loc[markerDF["Marker"].isin(posCorrs)]
            sns.barplot(data=ratioDF.tail(numFactors), y="Marker", x=distance, ax=ax[i], color='k')
            ax[i].set(xscale="log")
            ax[0].set(title="Wasserstein Distance - Surface Markers")
            ax[1].set(title="KL Divergence - Surface Markers")
        return corrsDF

    Wass_KL_Dist(ax[0],"NK", )
    
    return f