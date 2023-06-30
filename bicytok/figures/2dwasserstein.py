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
from sklearn.preprocessing import normalize
from ot import emd2



path_here = dirname(dirname(__file__))

def makeFigure():
    ax, f = getSetup((10, 8), (2, 2)) 
    def Wass_KL_Dist_2D(ax, targCell, numFactors, RNA=False, offTargState=0):
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

                # Combine receptor lists for 2D distribution calculation
                receptorList = np.column_stack((targCellMark, offTargCellMark))

                # Calculate Wasserstein distance using POT package
                distances = emd2(receptorList[:, 0], receptorList[:, 1], normalize(np.ones_like(receptorList[:, 0])), normalize(np.ones_like(receptorList[:, 1])))

                markerDF = pd.concat([markerDF, pd.DataFrame({"Marker": [marker], "Wasserstein Distance": distances})])

        corrsDF = pd.DataFrame()
        ratioDF = markerDF.sort_values(by="Wasserstein Distance")
        posCorrs = ratioDF.tail(numFactors).Marker.values
        corrsDF = pd.concat([corrsDF, pd.DataFrame({"Distance": "Wasserstein Distance", "Marker": posCorrs})])
        markerDF = markerDF.loc[markerDF["Marker"].isin(posCorrs)]
        sns.barplot(data=ratioDF.tail(numFactors), y="Marker", x="Wasserstein Distance", ax=ax, color='k')
        ax.set(xscale="log", title="Wasserstein Distance - Surface Markers")
        
        return corrsDF
    corrsDF = Wass_KL_Dist_2Dm(ax[0], targCell="Tregs", numFactors=5, offTargState=0)

#   Show the plot
    plt.show()
    return f