from os.path import dirname, join
from .common import getSetup, Wass_KL_Dist, correlation
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from ..imports import importCITE
from sklearn.neighbors import KernelDensity
from random import sample, seed
from scipy import stats

path_here = dirname(dirname(__file__))

plt.rcParams["svg.fonttype"] = "none"

def makeFigure():
    ax, f = getSetup((12, 4), (1, 2))

    CITE_DF = importCITE()
    new_df = CITE_DF.sample(10000, random_state=42)
    cellLevel = "CellType2"

    epitopesList = pd.read_csv(join(path_here, "data/epitopeList.csv"))
    epitopes = list(epitopesList['Epitope'].unique())

    cells = list(CITE_DF[cellLevel].unique())
    toRemove = ['Eryth', 'MAIT', 'ASDC', 'Platelet', 'gdT', 'dnT', 'B intermediate', 'CD4 CTL', 'NK Proliferating', 'CD8 Proliferating','CD4 Proliferating']
    for removed in toRemove:
        cells.remove(removed)

    df = pd.DataFrame(columns=['Cell', 'KL Divergence', "Earth Mover's Distance"])

    for targCell in cells:
        markerDF = pd.DataFrame(columns=["Marker", "KL", "EMD"])
        for marker in CITE_DF.loc[:, ((CITE_DF.columns != 'CellType1') & (CITE_DF.columns != 'CellType2') & (CITE_DF.columns != 'CellType3') & (CITE_DF.columns != 'Cell'))].columns:
            markAvg = np.mean(CITE_DF[marker].values)
            if markAvg > 0.0001:
                targCellMark = CITE_DF.loc[CITE_DF[cellLevel] == targCell][marker].values / markAvg
                offTargCellMark = CITE_DF.loc[CITE_DF[cellLevel] != targCell][marker].values / markAvg
                if np.mean(targCellMark) > np.mean(offTargCellMark):
                    kdeTarg = KernelDensity(kernel='gaussian').fit(targCellMark.reshape(-1, 1))
                    kdeOffTarg = KernelDensity(kernel='gaussian').fit(offTargCellMark.reshape(-1, 1))
                    minVal = np.minimum(targCellMark.min(), offTargCellMark.min()) - 10
                    maxVal = np.maximum(targCellMark.max(), offTargCellMark.max()) + 10
                    outcomes = np.arange(minVal, maxVal + 1).reshape(-1, 1)
                    distTarg = np.exp(kdeTarg.score_samples(outcomes))
                    distOffTarg = np.exp(kdeOffTarg.score_samples(outcomes))
                    KL_div = stats.entropy(distOffTarg.flatten() + 1e-200, distTarg.flatten() + 1e-200, base=2)
                    markerDF = pd.concat([markerDF, pd.DataFrame({"Marker": [marker], "KL": KL_div, "EMD": stats.wasserstein_distance(targCellMark, offTargCellMark)})], ignore_index=True)
        KLrow = markerDF.iloc[markerDF['KL'].idxmax()]
        markerKL = KLrow['Marker']
        KL = KLrow['KL']
        EMDrow = markerDF.iloc[markerDF['EMD'].idxmax()]
        markerEMD = EMDrow['Marker']
        EMD = EMDrow['EMD']

        data = {'Cell': targCell,
            'KL Marker': targCell + ' ' + markerKL,
            'KL Divergence': [KL],
            'EMD Marker': targCell + ' ' + markerEMD,
            "Earth Mover's Distance": [EMD]}
        df_temp = pd.DataFrame(data, columns=['Cell', 'KL Marker', 'KL Divergence', 'EMD Marker', "Earth Mover's Distance"])
        df = pd.concat([df, df_temp], ignore_index=True)

    sns.barplot(data=df.sort_values(by=['KL Divergence']), x='KL Marker', y='KL Divergence', ax=ax[0])
    sns.barplot(data=df.sort_values(by=["Earth Mover's Distance"]), x='EMD Marker', y="Earth Mover's Distance", ax=ax[1])
    ax[0].set_xticklabels(labels=ax[0].get_xticklabels(), rotation=45, horizontalalignment='right')
    ax[1].set_xticklabels(labels=ax[1].get_xticklabels(), rotation=45, horizontalalignment='right')

    return f