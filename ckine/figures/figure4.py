"""
This creates Figure 1, response of bispecific IL-2 cytokines at varing valencies and abundances using binding model.
"""
from .figureCommon import getSetup
from ..imports import importCITE, importReceptors
import pandas as pd
import seaborn as sns
import numpy as np
from copy import copy
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    ax, f = getSetup((8, 8), (2, 2))
    convFactCalc(ax[2])

    CITE_SVM(ax[0:2], "Treg", sampleFrac=0.2)

    return f


def CITE_SVM(ax, targCell, numFactors=10, sampleFrac=0.5):
    """Fits a ridge classifier to the CITE data and plots those most highly correlated with T reg"""
    SVMmod = SVC()
    SVC_DF = importCITE()
    cellToI = SVC_DF.CellType2.unique()
    SVC_DF = SVC_DF.loc[(SVC_DF["CellType2"].isin(cellToI)), :]
    SVC_DF = SVC_DF.sample(frac=sampleFrac, random_state=1)
    cellTypeCol = SVC_DF.CellType2.values
    SVC_DF = SVC_DF.loc[:, ((SVC_DF.columns != 'CellType1') & (SVC_DF.columns != 'CellType2') & (SVC_DF.columns != 'CellType3') & (SVC_DF.columns != 'Cell'))]
    factors = SVC_DF.columns
    X = SVC_DF.values
    X = StandardScaler().fit_transform(X)
    CD25col = X[:, np.where(factors == "CD25")].reshape(-1, 1)

    enc = LabelBinarizer()
    y = enc.fit_transform(cellTypeCol)
    TregY = y[:, np.where(enc.classes_ == targCell)].ravel()

    AccDF = pd.DataFrame(columns=["Markers", "Accuracy"])
    baselineAcc = SVMmod.fit(CD25col, TregY).score(CD25col, TregY)
    print(baselineAcc)
    print(np.where((factors == "CD25")))
    for marker in factors:
        SVMmod = SVC()
        print(marker)
        markerCol = X[:, np.where(factors == marker)]
        CD25MarkX = np.hstack((CD25col, markerCol.reshape(-1, 1)))
        markAcc = SVMmod.fit(CD25MarkX, TregY).score(CD25MarkX, TregY)
        print(markAcc)
        AccDF = AccDF.append(pd.DataFrame({"Markers": [marker], "Accuracy": [markAcc]}))

    AccDF = AccDF.sort_values(by="Accuracy")
    markers = copy(AccDF.tail(numFactors).Markers.values)  # Here
    AccDF.Markers = "CD25 + " + AccDF.Markers

    plot_DF = AccDF.tail(numFactors).append(pd.DataFrame({"Markers": ["CD25 only"], "Accuracy": [baselineAcc]}))
    sns.barplot(data=plot_DF, x="Markers", y="Accuracy", ax=ax[0])
    ax[0].set(ylim=(0.9, 1))
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45)

    SVC_DF = importCITE()
    markerDF = pd.DataFrame(columns=["Marker", "Cell Type", "Amount"])
    for marker in markers:
        for cell in cellToI:
            cellTDF = SVC_DF.loc[SVC_DF["CellType2"] == cell][marker]
            markerDF = markerDF.append(pd.DataFrame({"Marker": [marker], "Cell Type": cell, "Amount": cellTDF.mean(), "Number": cellTDF.size}))

    sns.pointplot(data=markerDF, x="Marker", y="Amount", hue="Cell Type", ax=ax[1], join=False, dodge=True)
    ax[1].set(yscale="log")
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45)


cellDict = {"CD4 Naive": "Thelper",
            "CD4 CTL": "Thelper",
            "CD4 TCM": "Thelper",
            "CD4 TEM": "Thelper",
            "NK": "NK",
            "CD8 Naive": "CD8",
            "CD8 TCM": "CD8",
            "CD8 TEM": "CD8",
            "Treg": "Treg"}


markDict = {"CD25": "IL2Ra",
            "CD122": "IL2Rb",
            "CD127": "IL7Ra",
            "CD132": "gc"}


def convFactCalc(ax):
    """Fits a ridge classifier to the CITE data and plots those most highly correlated with T reg"""
    CITE_DF = importCITE()
    cellToI = ["CD4 TCM", "CD8 Naive", "NK", "CD8 TEM", "CD4 Naive", "CD4 CTL", "CD8 TCM", "Treg", "CD4 TEM"]
    markers = ["CD122", "CD127", "CD25"]
    markerDF = pd.DataFrame(columns=["Marker", "Cell Type", "Amount", "Number"])
    for marker in markers:
        for cell in cellToI:
            cellTDF = CITE_DF.loc[CITE_DF["CellType2"] == cell][marker]
            markerDF = markerDF.append(pd.DataFrame({"Marker": [marker], "Cell Type": cell, "Amount": cellTDF.mean(), "Number": cellTDF.size}))

    markerDF = markerDF.replace({"Marker": markDict, "Cell Type": cellDict})
    markerDFw = pd.DataFrame(columns=["Marker", "Cell Type", "Average"])
    for marker in markerDF.Marker.unique():
        for cell in markerDF["Cell Type"].unique():
            subDF = markerDF.loc[(markerDF["Cell Type"] == cell) & (markerDF["Marker"] == marker)]
            wAvg = np.sum(subDF.Amount.values * subDF.Number.values) / np.sum(subDF.Number.values)
            markerDFw = markerDFw.append(pd.DataFrame({"Marker": [marker], "Cell Type": cell, "Average": wAvg}))

    recDF = importReceptors()
    weightDF = pd.DataFrame(columns=["Receptor", "Weight"])

    for rec in markerDFw.Marker.unique():
        CITEval = np.array([])
        Quantval = np.array([])
        for cell in markerDF["Cell Type"].unique():
            CITEval = np.concatenate((CITEval, markerDFw.loc[(markerDFw["Cell Type"] == cell) & (markerDFw["Marker"] == rec)].Average.values))
            Quantval = np.concatenate((Quantval, recDF.loc[(recDF["Cell Type"] == cell) & (recDF["Receptor"] == rec)].Mean.values))
        weightDF = weightDF.append(pd.DataFrame({"Receptor": [rec], "Weight": np.linalg.lstsq(np.reshape(CITEval, (-1, 1)), Quantval, rcond=None)[0]}))

    sns.barplot(data=weightDF, x="Receptor", y="Weight", ax=ax)
    ax.set(ylim=(0, 1000))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
