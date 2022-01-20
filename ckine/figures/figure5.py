"""
This creates Figure 5, used to find optimal epitope classifier.
"""
from email.mime import base
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
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer

path_here = dirname(dirname(__file__))


# Later can we test by making a fake cell with 0 or -1 for every value and verifying positions are maintained?
def makeFigure():
    """Get a list of the axis objects and create a figure"""
    ax, f = getSetup((9, 12), (2, 2))

    epitopesDF = pd.DataFrame(columns={"Classifier", "Epitope", "Selectivity"})
    epitopesDF = pd.read_csv(join(path_here, "data/epitopeList.csv"))

    CITE_DF = importCITE()

    # Get conv factors, average them
    convFact = convFactCalc(ax[0])
    meanConv = convFact.Weight.mean()

    # If selectivity values are stored in CSV at end of run
    saveFile = True

    # weighting idea: take sample of everything of ~3 times and then average each types amount and use that as the size

    cellList = CITE_DF["CellType2"].unique().tolist()
    print(cellList)

    sampleSizes = []
    for cellType in cellList:
        cellSample = []
        for i in np.arange(3):
            sampleDF = CITE_DF.sample(1000)
            sampleSize = int(len(sampleDF.loc[sampleDF["CellType2"] == cellType]))
            cellSample.append(sampleSize)
        meanSize = np.mean(cellSample)
        sampleSizes.append(int(meanSize))

    offTCells = cellList.copy()
    offTCells.remove('Treg')

    # For each  cellType in list
    for i, cellType in enumerate(cellList):

        # Generate sample size
        sampleSize = sampleSizes[i]

        cellDF = CITE_DF.loc[CITE_DF["CellType2"] == cellType].sample(sampleSize)

        cellType_abdundances = []
        # For each epitope (being done on per cell basis)
        for e in epitopesDF.Epitope:
            # calculate abundance based on converstion factor
            if e == 'CD25':
                convFact = 77.136987
            elif e == 'CD122':
                convFact = 332.680090
            elif e == "CD127":
                convFact = 594.379215
            else:
                convFact = meanConv

            citeVal = cellDF[e].to_numpy()
            abundance = citeVal * convFact
            cellType_abdundances.append(abundance)
            # add column with this name to epitopesDF and abundances list

        epitopesDF[cellType] = cellType_abdundances

    # EpitopeDF now contains a data of single cell abundances for each cell type for each epitope
    epitopesDF['Selectivity'] = -1
    # New column which will hold selectivity per epitope

    targCell = 'Treg'

    # Feed actual abundance into modeling
    abundanceDF = pd.DataFrame(columns={"Receptor", "Cell Type", "Abundance"})

    standardDF = epitopesDF.loc[(epitopesDF.Epitope == 'CD25')].sample()
    standard2DF = epitopesDF.loc[(epitopesDF.Epitope == 'CD122')].sample()
    standardDF = standardDF.append(standard2DF)
    standardDF['Type'] = 'Standard'
    # For each epitope

    for epitope in epitopesDF['Epitope'].unique():
        # print(epitope)

        selectedDF = epitopesDF.loc[(epitopesDF.Epitope == epitope)].sample()
        selectedDF['Type'] = 'Epitope'
        selectedDF = selectedDF.append(standardDF)
        selectedDF.reset_index()

        # New form
        optSelectivity = 1 / (optimizeDesign(targCell, offTCells, selectedDF, epitope))

        # print(optSelectivity)
        epitopesDF.loc[epitopesDF['Epitope'] == epitope, 'Selectivity'] = optSelectivity  # Store selectivity in DF to be used for plots

    baseSelectivity = 1 / (selecCalc(standardDF, targCell, offTCells))

    if saveFile:
        epitopesDF = epitopesDF[["Classifier", "Epitope", "Selectivity"]]  # drops single cell info
        epitopesDF.to_csv(join(path_here, "data/epitopeSelectivityList.csv"), index=False)
        print("File Saved")

    # generate figures

    classifiers = ['CITE_SVM', 'CITE_RIDGE', 'distMetricF', 'distMetricT']
    # for each classifier
    for i, classifier in enumerate(classifiers):
        # bar plot of each epitope
        print(i)
        epitopesDF = epitopesDF.sort_values(by=['Selectivity'])
        xvalues = epitopesDF.loc[epitopesDF['Classifier'] == classifier, 'Epitope']
        yvalues = (((epitopesDF.loc[epitopesDF['Classifier'] == classifier, 'Selectivity']) / baseSelectivity) * 100) - 100
        print(yvalues)
        cmap = sns.color_palette("husl", 10)
        sns.barplot(x=xvalues, y=yvalues, palette=cmap, ax=ax[i]).set_title(classifier)
        ax[i].set_ylabel("Selectivity (% increase over standard IL2)")

        # change color scheme
        # order bars

        # standardize y axis across plots

    return f


def cytBindingModel(counts, x=False, date=False):
    """Runs binding model for a given mutein, valency, dose, and cell type."""
    mut = 'IL2'
    val = 1
    doseVec = np.array([0.1])

    #
    recCount = np.ravel(counts)
    #

    mutAffDF = pd.read_csv(join(path_here, "data/WTmutAffData.csv"))
    Affs = mutAffDF.loc[(mutAffDF.Mutein == mut)]
    Affs = np.power(np.array([Affs["IL2RaKD"].values, Affs["IL2RBGKD"].values]) / 1e9, -1)
    Affs = np.reshape(Affs, (1, -1))
    Affs = np.repeat(Affs, 2, axis=0)
    np.fill_diagonal(Affs, 1e2)  # Each cytokine can only bind one a and one b

    if doseVec.size == 1:
        doseVec = np.array([doseVec])
    output = np.zeros(doseVec.size)

    for i, dose in enumerate(doseVec):
        if x:
            output[i] = polyc(dose / 1e9, np.power(10, x[0]), recCount, [[val, val]], [1.0], Affs)[0][1]
        else:
            output[i] = polyc(dose / 1e9, getKxStar(), recCount, [[val, val]], [1.0], Affs)[0][1]  # IL2RB binding only

    return output


def cytBindingModel_bispecOpt(counts, recXaff, x=False):
    """Runs binding model for a given mutein, valency, dose, and cell type."""

    mut = 'IL2'
    val = 1
    doseVec = np.array([0.1])

    recXaff = np.power(10, recXaff)

    recCount = np.ravel(counts)

    mutAffDF = pd.read_csv(join(path_here, "data/WTmutAffData.csv"))
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

    return output


def selecCalc(df, targCell, offTCells):
    """Calculates selectivity for no additional epitope"""
    targetBound = 0
    offTargetBound = 0

    cd25DF = df.loc[(df.Type == 'Standard') & (df.Epitope == 'CD25')]
    cd122DF = df.loc[(df.Type == 'Standard') & (df.Epitope == 'CD122')]

    for i, cd25Count in enumerate(cd25DF[targCell].item()):
        cd122Count = cd122DF[targCell].item()[i]
        counts = [cd25Count, cd122Count]
        targetBound += cytBindingModel(counts)
    for cellT in offTCells:
        for i, cd25Count in enumerate(cd25DF[cellT].item()):
            cd122Count = cd122DF[cellT].item()[i]
            counts = [cd25Count, cd122Count]
            offTargetBound += cytBindingModel(counts)

    return (offTargetBound) / (targetBound)


def minSelecFunc(x, selectedDF, targCell, offTCells, epitope):
    """Provides the function to be minimized to get optimal selectivity"""
    targetBound = 0
    offTargetBound = 0

    recXaff = x

    epitopeDF = selectedDF.loc[(selectedDF.Type == 'Epitope')]
    cd25DF = selectedDF.loc[(selectedDF.Type == 'Standard') & (selectedDF.Epitope == 'CD25')]
    cd122DF = selectedDF.loc[(selectedDF.Type == 'Standard') & (selectedDF.Epitope == 'CD122')]

    for i, epCount in enumerate(epitopeDF[targCell].item()):
        cd25Count = cd25DF[targCell].item()[i]
        cd122Count = cd122DF[targCell].item()[i]
        counts = [cd25Count, cd122Count, epCount]
        targetBound += cytBindingModel_bispecOpt(counts, recXaff)
    for cellT in offTCells:
        for i, epCount in enumerate(epitopeDF[cellT].item()):
            cd25Count = cd25DF[cellT].item()[i]
            cd122Count = cd122DF[cellT].item()[i]
            counts = [cd25Count, cd122Count, epCount]

            offTargetBound += cytBindingModel_bispecOpt(counts, recXaff)

    return (offTargetBound) / (targetBound)


def optimizeDesign(targCell, offTcells, selectedDF, epitope):
    """ A more general purpose optimizer """
    ###
    #vals = np.arange(1.01, 10, step=0.15)
    #sigDF = pd.DataFrame()
    #optDF = pd.DataFrame(columns={"Valency", "Selectivity", "IL2Rα", r"IL-2Rβ/γ$_c$"})
    ###

    if targCell == "NK":
        X0 = [6.0, 8]
    else:
        X0 = [7.0]

    optBnds = Bounds(np.full_like(X0, 6.0), np.full_like(X0, 9.0))

    optimized = minimize(minSelecFunc, X0, bounds=optBnds, args=(selectedDF, targCell, offTcells, epitope), jac="3-point")
    optSelectivity = optimized.fun[0]

    return optSelectivity


def CITE_SVM(ax, targCell, numFactors=10, sampleFrac=0.2):
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
    # print(baselineAcc)
    #print(np.where((factors == "CD25")))
    for marker in factors:
        SVMmod = SVC()
        # print(marker)
        markerCol = X[:, np.where(factors == marker)]
        CD25MarkX = np.hstack((CD25col, markerCol.reshape(-1, 1)))
        markAcc = SVMmod.fit(CD25MarkX, TregY).score(CD25MarkX, TregY)
        # print(markAcc)
        AccDF = AccDF.append(pd.DataFrame({"Markers": [marker], "Accuracy": [markAcc]}))

    AccDF = AccDF.sort_values(by="Accuracy")
    markers = copy(AccDF.tail(numFactors).Markers.values)  # Here
    AccDF.Markers = "CD25 + " + AccDF.Markers

    print(markers)
    return markers


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

    return weightDF
