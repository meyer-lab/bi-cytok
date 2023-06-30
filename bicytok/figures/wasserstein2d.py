import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KernelDensity
from scipy import stats
from ot import emd2_samples
from ot.lp import emd2
from ot.datasets import make_1D_gauss as gauss
from os.path import dirname, join
from .common import getSetup
from ..imports import importCITE
import matplotlib.pyplot as plt

def makeFigure():
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    def Wass_KL_Dist2d(ax, targCell, numFactors, offTargReceptors, signalReceptor, RNA=False, offTargState=0):
        """Finds markers which have average greatest difference from other cells"""
        CITE_DF = importCITE()

        markerDF = pd.DataFrame(columns=["Marker", "Cell Type", "Amount"])
        for marker in CITE_DF.loc[:, ((CITE_DF.columns != 'CellType1') & (CITE_DF.columns != 'CellType2') & (CITE_DF.columns != 'CellType3') & (CITE_DF.columns != 'Cell'))].columns:
            markAvg = np.mean(CITE_DF[marker].values)
            if markAvg > 0.0001:
                targCellMark = np.vstack((CITE_DF.loc[CITE_DF["CellType3"] == targCell][offTargReceptors[0]].values,
                                        CITE_DF.loc[CITE_DF["CellType3"] == targCell][signalReceptor].values)).T
                offTargCellMark = np.vstack((CITE_DF.loc[CITE_DF["CellType3"] != targCell][offTargReceptors[0]].values,
                                            CITE_DF.loc[CITE_DF["CellType3"] != targCell][signalReceptor].values)).T
                Wass_dist = emd2_samples(targCellMark, offTargCellMark)
                markerDF = pd.concat([markerDF, pd.DataFrame({"Marker": [marker], "Wasserstein Distance": Wass_dist})])

        corrsDF = pd.DataFrame()
        for i, distance in enumerate(["Wasserstein Distance"]):
            ratioDF = markerDF.sort_values(by=distance)
            posCorrs = ratioDF.tail(numFactors).Marker.values
            corrsDF = pd.concat([corrsDF, pd.DataFrame({"Distance": distance, "Marker": posCorrs})])
            markerDF = markerDF.loc[markerDF["Marker"].isin(posCorrs)]
            sns.barplot(data=ratioDF.tail(numFactors), y="Marker", x=distance, ax=ax[i], color='k')
            ax[i].set(xscale="log")
            ax[0].set(title="Wasserstein Distance - Receptor Space")
        return corrsDF
    
    targCell = "Treg"
    numFactors = 5
    offTargReceptors = ["CD335"]  # Update with the list of off-target receptors
    signalReceptor = "IL2RB"  # Update with the signaling receptor
    
    result = Wass_KL_Dist2d(ax, targCell, numFactors, offTargReceptors, signalReceptor)

    # Display the bar plots
    plt.show()

    # Print the resulting DataFrame
    print(result)
    return fig
