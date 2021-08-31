"""
This creates Figure 1, response of bispecific IL-2 cytokines at varing valencies and abundances using binding model.
"""
from .figureCommon import getSetup
from os.path import join
from ..imports import importCITE
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import LabelEncoder


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    ax, f = getSetup((15, 12), (3, 2), multz={4:1})
    posCorrs, negCorrs = CITE_RIDGE(ax[3])
    RIDGE_Scatter(ax[4], posCorrs, negCorrs)
    CITE_PCA(ax[0:3], posCorrs, negCorrs)
    
    return f

def CITE_PCA(ax, posCorrs, negCorrs):
    """ Plots all surface markers in PCA format"""
    PCA_DF = importCITE()
    cellToI = ["CD4 TCM", "CD8 Naive", "NK", "CD8 TEM", "CD4 Naive", "CD4 CTL", "CD8 TCM", "Treg", "CD4 TEM", "NK_CD56bright"]
    PCA_DF = PCA_DF.loc[(PCA_DF["CellType2"].isin(cellToI)), :]
    cellType = PCA_DF.CellType2.values
    PCA_DF = PCA_DF.loc[:, ((PCA_DF.columns != 'CellType1') & (PCA_DF.columns != 'CellType2') & (PCA_DF.columns != 'CellType3') & (PCA_DF.columns != 'Cell'))]
    factors = PCA_DF.columns
    scaler = StandardScaler()

    pca = PCA(n_components=2)
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
    sns.scatterplot(data=scoresDF, x="PC 1", y="PC 2", ax=ax[0], hue="Cell Type", alpha=0.3)
    ax[0].set(xlim=(-20, 50), ylim=(-50, 50))

    centerDF = pd.DataFrame(columns=["PC 1", "PC 2", "Cell Type"])
    for cell in cellToI:
        cellTDF = scoresDF.loc[scoresDF["Cell Type"] == cell]
        centerDF = centerDF.append(pd.DataFrame({"PC 1": [cellTDF["PC 1"].mean()], "PC 2": [cellTDF["PC 2"].mean()], "Cell Type": [cell]}))

    sns.scatterplot(data=centerDF, x="PC 1", y="PC 2", ax=ax[2], hue="Cell Type")
    ax[0].set(xlim=(-20, 50), ylim=(-50, 50))

def CITE_RIDGE(ax, numFactors=10):
    """Fits a ridge classifier to the CITE data and plots those most highly correlated with T reg"""
    ridgeMod = RidgeClassifierCV()
    RIDGE_DF = importCITE()
    cellToI = ["CD4 TCM", "CD8 Naive", "NK", "CD8 TEM", "CD4 Naive", "CD4 CTL", "CD8 TCM", "Treg", "CD4 TEM", "NK_CD56bright"]
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
    TregCoefs = ridgeMod.coef_[np.where(le.classes_=="Treg"), :].ravel()
    TregCoefsDF = pd.DataFrame({"Marker": factors, "Coefficient": TregCoefs}).sort_values(by="Coefficient")
    TregCoefsDF = pd.concat([TregCoefsDF.head(numFactors), TregCoefsDF.tail(numFactors)])
    sns.barplot(data=TregCoefsDF, x="Marker", y="Coefficient", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    posCorrs = TregCoefsDF.tail(numFactors).Marker.values
    negCorrs = TregCoefsDF.head(numFactors).Marker.values

    return posCorrs, negCorrs


def RIDGE_Scatter(ax, posCorrs, negCorrs):
    """Fits a ridge classifier to the CITE data and plots those most highly correlated with T reg"""
    CITE_DF = importCITE()
    cellToI = ["CD4 TCM", "CD8 Naive", "NK", "CD8 TEM", "CD4 Naive", "CD4 CTL", "CD8 TCM", "Treg", "CD4 TEM", "NK_CD56bright"]
    CITE_DF = CITE_DF.loc[(CITE_DF["CellType2"].isin(cellToI)), :]
    cellTypeCol = CITE_DF.CellType2.values

    CITE_DF = CITE_DF[np.append(negCorrs, posCorrs)]
    CITE_DF["Cell Type"] = cellTypeCol
    CITE_DF = pd.melt(CITE_DF, id_vars="Cell Type", var_name="Marker", value_name='Amount')

    sns.pointplot(data=CITE_DF, x="Marker", y="Amount", hue="Cell Type", ax=ax, join=False, dodge=True)
    ax.set(yscale="log")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
