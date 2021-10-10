"""
This creates Figure 1, response of bispecific IL-2 cytokines at varing valencies and abundances using binding model.
"""
from .figureCommon import getSetup
from ..imports import importCITE
from sklearn.decomposition import PCA
from copy import copy
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    ax, f = getSetup((9, 12), (5, 2))
    cellTarget = "Treg"
    ax[1].axis("off")

    CITE_TSNE(ax[0], sampleFrac=0.4)
    legend = ax[0].get_legend()
    labels = (x.get_text() for x in legend.get_texts())
    ax[1].legend(legend.legendHandles, labels, loc="upper left", prop={"size": 10}, ncol=3)
    ax[0].get_legend().remove()

    posCorrs, negCorrs = CITE_RIDGE(ax[4], cellTarget)
    CITE_PCA(ax[2:4], posCorrs, negCorrs)
    RIDGE_Scatter(ax[5], posCorrs, negCorrs)
    distMetricScatt(ax[6:8], cellTarget, 10, weight=False)
    distMetricScatt(ax[8:10], cellTarget, 10, weight=True)

    return f


def CITE_TSNE(ax, sampleFrac):
    """ Plots all surface markers in PCA format"""
    TSNE_DF = importCITE()
    cellToI = TSNE_DF.CellType2.unique()
    TSNE_DF = TSNE_DF.loc[(TSNE_DF["CellType2"].isin(cellToI)), :]
    TSNE_DF = TSNE_DF.sample(frac=sampleFrac, random_state=1)
    cellType = TSNE_DF.CellType2.values
    TSNE_DF = TSNE_DF.loc[:, ((TSNE_DF.columns != 'CellType1') & (TSNE_DF.columns != 'CellType2') & (TSNE_DF.columns != 'CellType3') & (TSNE_DF.columns != 'Cell'))]
    factors = TSNE_DF.columns
    scaler = StandardScaler()

    TSNE_Arr = scaler.fit_transform(X=TSNE_DF.values)
    pca = PCA(n_components=20)
    TSNE_PCA_Arr = pca.fit_transform(TSNE_Arr)

    X_embedded = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000, learning_rate=200).fit_transform(TSNE_PCA_Arr)
    df_tsne = pd.DataFrame(X_embedded, columns=['comp1', 'comp2'])
    df_tsne['label'] = cellType
    sns.scatterplot(x='comp1', y='comp2', data=df_tsne, hue='label', alpha=0.3, ax=ax, s=2)


def CITE_PCA(ax, posCorrs, negCorrs):
    """ Plots all surface markers in PCA format"""
    PCA_DF = importCITE()
    cellToI = PCA_DF.CellType2.unique()
    PCA_DF = PCA_DF.loc[(PCA_DF["CellType2"].isin(cellToI)), :]
    cellType = PCA_DF.CellType2.values
    PCA_DF = PCA_DF.loc[:, ((PCA_DF.columns != 'CellType1') & (PCA_DF.columns != 'CellType2') & (PCA_DF.columns != 'CellType3') & (PCA_DF.columns != 'Cell'))]
    factors = PCA_DF.columns
    scaler = StandardScaler()

    pca = PCA(n_components=10)
    PCA_Arr = scaler.fit_transform(X=PCA_DF.values)
    pca.fit(PCA_Arr)
    comps = pca.components_
    loadingsDF = pd.DataFrame({"PC 1": comps[0], "PC 2": comps[1], "Factors": factors})
    loadP = sns.scatterplot(data=loadingsDF, x="PC 1", y="PC 2", ax=ax[1])

    for line in range(0, loadingsDF.shape[0]):
        if factors[line] in posCorrs:
            loadP.text(loadingsDF["PC 1"][line] + 0.002, loadingsDF["PC 2"][line],
                       factors[line], horizontalalignment='left',
                       size='medium', color='green', weight='semibold')
        if factors[line] in negCorrs:
            loadP.text(loadingsDF["PC 1"][line] + 0.002, loadingsDF["PC 2"][line],
                       factors[line], horizontalalignment='left',
                       size='medium', color='red', weight='semibold')

    scores = pca.transform(PCA_Arr)
    scoresDF = pd.DataFrame({"PC 1": scores[:, 0], "PC 2": scores[:, 1], "Cell Type": cellType})

    centerDF = pd.DataFrame(columns=["PC 1", "PC 2", "Cell Type"])
    for cell in cellToI:
        cellTDF = scoresDF.loc[scoresDF["Cell Type"] == cell]
        centerDF = centerDF.append(pd.DataFrame({"PC 1": [cellTDF["PC 1"].mean()], "PC 2": [cellTDF["PC 2"].mean()], "Cell Type": [cell]}))

    sns.scatterplot(data=centerDF, x="PC 1", y="PC 2", ax=ax[0], hue="Cell Type", legend=False)
    ax[0].set(xlim=(-10, 30), ylim=(-5, 5))


def CITE_RIDGE(ax, targCell, numFactors=10):
    """Fits a ridge classifier to the CITE data and plots those most highly correlated with T reg"""
    ridgeMod = RidgeClassifierCV()
    RIDGE_DF = importCITE()
    cellToI = RIDGE_DF.CellType2.unique()
    RIDGE_DF = RIDGE_DF.loc[(RIDGE_DF["CellType2"].isin(cellToI)), :]
    cellTypeCol = RIDGE_DF.CellType2.values
    RIDGE_DF = RIDGE_DF.loc[:, ((RIDGE_DF.columns != 'CellType1') & (RIDGE_DF.columns != 'CellType2') & (RIDGE_DF.columns != 'CellType3') & (RIDGE_DF.columns != 'Cell'))]
    factors = RIDGE_DF.columns
    X = RIDGE_DF.values
    X = StandardScaler().fit_transform(X)

    le = LabelEncoder()
    le.fit(cellTypeCol)
    y = le.transform(cellTypeCol)

    ridgeMod = RidgeClassifierCV(cv=5)
    ridgeMod.fit(X, y)
    TargCoefs = ridgeMod.coef_[np.where(le.classes_ == targCell), :].ravel()
    TargCoefsDF = pd.DataFrame({"Marker": factors, "Coefficient": TargCoefs}).sort_values(by="Coefficient")
    TargCoefsDF = pd.concat([TargCoefsDF.head(numFactors), TargCoefsDF.tail(numFactors)])
    sns.barplot(data=TargCoefsDF, x="Marker", y="Coefficient", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    posCorrs = TargCoefsDF.tail(numFactors).Marker.values
    negCorrs = TargCoefsDF.head(numFactors).Marker.values

    print(posCorrs)
    return posCorrs, negCorrs


def RIDGE_Scatter(ax, posCorrs, negCorrs):
    """Fits a ridge classifier to the CITE data and plots those most highly correlated with T reg"""
    CITE_DF = importCITE()
    cellToI = CITE_DF.CellType2.unique()
    CITE_DF = CITE_DF.loc[(CITE_DF["CellType2"].isin(cellToI)), :]
    cellTypeCol = CITE_DF.CellType2.values

    CITE_DF = CITE_DF[np.append(negCorrs, posCorrs)]
    CITE_DF["Cell Type"] = cellTypeCol
    CITE_DF = pd.melt(CITE_DF, id_vars="Cell Type", var_name="Marker", value_name='Amount')

    sns.pointplot(data=CITE_DF, x="Marker", y="Amount", hue="Cell Type", ax=ax, join=False, dodge=True, legend=False)
    ax.set(yscale="log")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.get_legend().remove()


def distMetricScatt(ax, targCell, numFactors, weight=False):
    """Finds markers which have average greatest difference from other cells"""
    CITE_DF = importCITE()
    cellToI = CITE_DF.CellType2.unique()
    offTargs = copy(cellToI)
    offTargs = np.delete(offTargs, np.where(offTargs == targCell))
    CITE_DF = CITE_DF.loc[(CITE_DF["CellType2"].isin(cellToI)), :]
    cellTypeCol = CITE_DF.CellType2.values

    markerDF = pd.DataFrame(columns=["Marker", "Cell Type", "Amount"])
    for marker in CITE_DF.loc[:, ((CITE_DF.columns != 'CellType1') & (CITE_DF.columns != 'CellType2') & (CITE_DF.columns != 'CellType3') & (CITE_DF.columns != 'Cell'))].columns:
        for cell in cellToI:
            cellTDF = CITE_DF.loc[CITE_DF["CellType2"] == cell][marker]
            markerDF = markerDF.append(pd.DataFrame({"Marker": [marker], "Cell Type": cell, "Amount": cellTDF.mean(), "Number": cellTDF.size}))

    ratioDF = pd.DataFrame(columns=["Marker", "Ratio"])
    for marker in CITE_DF.loc[:, ((CITE_DF.columns != 'CellType1') & (CITE_DF.columns != 'CellType2') & (CITE_DF.columns != 'CellType3') & (CITE_DF.columns != 'Cell'))].columns:
        if weight:
            offT = 0
            targ = markerDF.loc[(markerDF["Cell Type"] == targCell) & (markerDF["Marker"] == marker)].Amount.mean()
            for cell in offTargs:
                offT += markerDF.loc[(markerDF["Cell Type"] == cell) & (markerDF["Marker"] == marker)].Amount.mean()
            ratioDF = ratioDF.append(pd.DataFrame({"Marker": [marker], "Ratio": (targ * len(offTargs)) / offT}))
        else:
            offT = 0
            targ = markerDF.loc[(markerDF["Cell Type"] == targCell) & (markerDF["Marker"] == marker)].Amount.values * \
                markerDF.loc[(markerDF["Cell Type"] == targCell) & (markerDF["Marker"] == marker)].Number.values
            for cell in offTargs:
                offT += markerDF.loc[(markerDF["Cell Type"] == cell) & (markerDF["Marker"] == marker)].Amount.values * \
                    markerDF.loc[(markerDF["Cell Type"] == cell) & (markerDF["Marker"] == marker)].Number.values
            ratioDF = ratioDF.append(pd.DataFrame({"Marker": [marker], "Ratio": (targ * len(offTargs)) / offT}))

    ratioDF = ratioDF.sort_values(by="Ratio")
    posCorrs = ratioDF.tail(numFactors).Marker.values

    markerDF = markerDF.loc[markerDF["Marker"].isin(posCorrs)]

    sns.barplot(data=ratioDF.tail(numFactors), x="Marker", y="Ratio", ax=ax[0])
    ax[0].set(yscale="log")
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45)
    if weight:
        ax[0].set(title="Ratios Weighted by Cell Type")
    else:
        ax[0].set(title="Ratios Weighted by Number of Cells")

    sns.pointplot(data=markerDF, x="Marker", y="Amount", hue="Cell Type", ax=ax[1], join=False, dodge=True, order=posCorrs, legend=False)
    ax[1].set(yscale="log")
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45)
    if weight:
        ax[1].set(title="Markers Weighted by Cell Type")
    else:
        ax[1].set(title="Markers Weighted by Number of Cells")
    ax[1].get_legend().remove()
