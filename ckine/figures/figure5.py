"""
This creates Figure 5, used to find optimal epitope classifier.
"""
from os.path import dirname, join
from .figureCommon import getSetup
from ..imports import importCITE, importReceptors, getBindDict
from ..MBmodel import polyc, getKxStar
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from copy import copy
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import LabelEncoder

path_here = dirname(dirname(__file__))

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    ax, f = getSetup((9, 12), (5, 2))
    cellTarget = "Treg"
    epitopesDF = pd.DataFrame(columns={"Classifier", "Epitope"})

    posCorrs1, negCorrs = CITE_RIDGE(ax[4], cellTarget)
    for x in posCorrs1:
        epitopesDF = epitopesDF.append(pd.DataFrame({"Classifier": 'CITE_RIDGE', "Epitope": [x]}))
    possCorrs2 = distMetricScatt(ax[6:8], cellTarget, 10, weight=False)
    for x in possCorrs2:
        epitopesDF = epitopesDF.append(pd.DataFrame({"Classifier": 'distMetricF', "Epitope": [x]}))
    possCorrs3 = distMetricScatt(ax[8:10], cellTarget, 10, weight=True)
    for x in possCorrs3:
        epitopesDF = epitopesDF.append(pd.DataFrame({"Classifier": 'distMetricT', "Epitope": [x]}))
    print(epitopesDF)
    #do for Cite_SVM

    #put these three in data frame, get abundance and affinity data

    #use minSelect function 
        #Feed into bispec binding model
    
    #optimize using minSelect



    return f


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
    #sns.barplot(data=TargCoefsDF, x="Marker", y="Coefficient", ax=ax)
    #ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    posCorrs = TargCoefsDF.tail(numFactors).Marker.values
    negCorrs = TargCoefsDF.head(numFactors).Marker.values

    return posCorrs, negCorrs

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
    return(posCorrs)

def cytBindingModel_bispecOpt(recX, recXaff, cellType, x=False):
    """Runs binding model for a given mutein, valency, dose, and cell type."""

    mut = 'IL2'
    val = 1
    doseVec = [0.1]
    date = '3/15/2019'


    recDF = importReceptors()
    recCount = np.ravel([recDF.loc[(recDF.Receptor == "IL2Ra") & (recDF["Cell Type"] == cellType)].Mean.values[0],
                         recDF.loc[(recDF.Receptor == "IL2Rb") & (recDF["Cell Type"] == cellType)].Mean.values[0], recX])

    mutAffDF = pd.read_csv(join(path_here, "ckine/data/WTmutAffData.csv"))
    Affs = mutAffDF.loc[(mutAffDF.Mutein == mut)]
    Affs = np.power(np.array([Affs["IL2RaKD"].values, Affs["IL2RBGKD"].values]) / 1e9, -1)
    Affs = np.reshape(Affs, (1, -1))
    Affs = np.append(Affs, recXaff)
    holder = np.full((3, 3), 1e2)
    np.fill_diagonal(holder, Affs)
    Affs = holder

    # Check that values are in correct placement, can invert

    if doseVec.size == 1:
        doseVec = np.array([doseVec])
    output = np.zeros(doseVec.size)

    for i, dose in enumerate(doseVec):
        if x:
            output[i] = polyc(dose / (val * 1e9), np.power(10, x[0]), recCount, [[val, val, val]], [1.0], Affs)[0][1]
        else:
            output[i] = polyc(dose / (val * 1e9), getKxStar(), recCount, [[val, val, val]], [1.0], Affs)[0][1]  # IL2RB binding only
    if date:
        convDict = getBindDict()
        if cellType[-1] == "$":  # if it is a binned pop, use ave fit
            output *= convDict.loc[(convDict.Date == date) & (convDict.Cell == cellType[0:-13])].Scale.values
        else:
            output *= convDict.loc[(convDict.Date == date) & (convDict.Cell == cellType)].Scale.values
    return output

def minSelecFunc(x, targCell, offTCells):
    """Provides the function to be minimized to get optimal selectivity"""
    offTargetBound = 0

    targCell = 'Treg'
    offTCells = ['Thelper','CD8','NK']

    recX = 6000
    recXaff = x

    targetBound = cytBindingModel_bispecOpt(recX, recXaff,targCell)
    for cellT in offTCells:
        offTargetBound += cytBindingModel_bispecOpt(recX, recXaff, cellT)

    return (offTargetBound) / (targetBound)

def optimizeDesign(ax, targCell, offTcells, legend=True):
    """ A more general purpose optimizer """
    vals = np.arange(1.01, 10, step=0.15)
    sigDF = pd.DataFrame()

    recX=6000

    
    optDF = pd.DataFrame(columns={"Valency", "Selectivity", "IL2Rα", r"IL-2Rβ/γ$_c$"})
    if targCell[0] == "NK":
        X0 = [6.0, 8]  # IL2Ra, IL2Rb
    else:
        X0 = [9.0, 6.0]  # IL2Ra, IL2Rb

    optBnds = Bounds(np.full_like(X0, 6.0), np.full_like(X0, 9.0))

    for i, val in enumerate(vals):
        if i == 0:
            optimized = minimize(minSelecFunc, X0, bounds=optBnds, args=(targCell, offTcells), jac="3-point")

            targLB = cytBindingModel_bispecOpt(recX, optimized.x, targCell[0]) / 1.01
            bindConst = NonlinearConstraint(lambda x: cytBindingModel_bispecOpt(recX, x, targCell[0]), targLB, np.inf)
        else:
            optimized = minimize(minSelecFunc, X0, bounds=optBnds, args=(val, targCell, offTcells), jac="3-point", constraints=bindConst)

        fitX = 1.0e9 / np.power(10.0, optimized.x)

        optDF = optDF.append(pd.DataFrame({"Valency": [val], "Selectivity": [len(offTcells) / optimized.fun], "IL2Rα": fitX[0], r"IL-2Rβ/γ$_c$": fitX[1]}))
        sigDF = sigDF.append(pd.DataFrame({"Cell Type": [targCell[0]], "Target": ["Target"], "Valency": [val], "pSTAT": [cytBindingModel_bispecOpt(recX, optimized.x, targCell[0])]}))
        for cell in offTcells:
            sigDF = sigDF.append(pd.DataFrame({"Cell Type": [cell], "Target": ["Off-Target"], "Valency": [val], "pSTAT": [cytBindingModel_bispecOpt(recX, optimized.x, cell)]}))
    # Normalize to valency 1
    for cell in targCell + offTcells:
        sigDF.loc[sigDF["Cell Type"] == cell, "pSTAT"] = sigDF.loc[sigDF["Cell Type"] == cell, "pSTAT"].div(sigDF.loc[(sigDF["Cell Type"] == cell) & (sigDF.Valency == vals[0])].pSTAT.values[0])

    # sigDF = sigDF.replace(cellTypeDict)


    #sns.lineplot(x="Valency", y="pSTAT", hue="Cell Type", style="Target", data=sigDF, ax=ax[0], palette="husl", hue_order=cellTypeDict.values())
    #ax[0].set_title(cellTypeDict[targCell[0]] + " selectivity with IL-2 mutein", fontsize=7)

    # if targCell[0] == "NK":
    #     affDF = pd.melt(optDF, id_vars=['Valency'], value_vars=[r"IL-2Rβ/γ$_c$"])
    #     sns.lineplot(x="Valency", y="value", data=affDF, ax=ax[1])
    #     ax[1].set(yscale="log", ylabel=r"IL2·β/γ$_c$ K$_D$ (nM)")
    # else:
    #     affDF = pd.melt(optDF, id_vars=['Valency'], value_vars=['IL2Rα', r"IL-2Rβ/γ$_c$"])
    #     affDF = affDF.rename(columns={"variable": "Receptor"})
    #     sns.lineplot(x="Valency", y="value", hue="Receptor", data=affDF, ax=ax[1])
    #     ax[1].set(yscale="log", ylabel=r"IL2· $K_D$ (nM)")

    # ax[0].set_ylim(bottom=0.0, top=3)
    # ax[1].set_ylim(bottom=0.1, top=2000)

    # ax[0].set_xticks(np.arange(1, 11))
    # ax[1].set_xticks(np.arange(1, 11))
    # if not legend:
    #     ax[0].get_legend().remove()