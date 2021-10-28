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
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer

path_here = dirname(dirname(__file__))

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    ax, f = getSetup((9, 12), (5, 2))
    cellTarget = "Treg"
    epitopesDF = pd.DataFrame(columns={"Classifier", "Epitope", "Selectivity"})

    # possCorrs0 = CITE_SVM(ax[0], cellTarget)
    # for x in possCorrs0:
    #     epitopesDF = epitopesDF.append(pd.DataFrame({"Classifier": 'CITE_SVM', "Epitope": [x]}))
    # print(epitopesDF)
    # posCorrs1, negCorrs = CITE_RIDGE(ax[4], cellTarget)
    # for x in posCorrs1:
    #     epitopesDF = epitopesDF.append(pd.DataFrame({"Classifier": 'CITE_RIDGE', "Epitope": [x]}))
    # possCorrs2 = distMetricScatt(ax[6:8], cellTarget, 10, weight=False)
    # for x in possCorrs2:
    #     epitopesDF = epitopesDF.append(pd.DataFrame({"Classifier": 'distMetricF', "Epitope": [x]}))
    # possCorrs3 = distMetricScatt(ax[8:10], cellTarget, 10, weight=True)
    # for x in possCorrs3:
    #     epitopesDF = epitopesDF.append(pd.DataFrame({"Classifier": 'distMetricT', "Epitope": [x]}))
    # print(epitopesDF)
    
    # epitopesDF.to_csv(join(path_here, "data/epitopeList.csv"), index=False)

    # Comment out running stuff
    # Import existing data frame
    # Import cite data
    epitopesDF = pd.read_csv(join(path_here, "data/epitopeList.csv"))

    CITE_DF = importCITE()

    # Get conv factors, average them
    convFact = convFactCalc(ax[0])
    meanConv = convFact.Weight.mean()
    

    # Import cite data into dataframe
    tregDF = CITE_DF.loc[CITE_DF["CellType2"] == 'Treg'].sample(100)
    nkDF = CITE_DF.loc[CITE_DF["CellType1"] == 'NK'].sample(100)
    
    thelperDF = CITE_DF.loc[CITE_DF["CellType2"]=='CD4 Naive']
    thelperDF = thelperDF.append(CITE_DF.loc[CITE_DF["CellType2"]=='CD4 CTL'])
    thelperDF = thelperDF.append(CITE_DF.loc[CITE_DF["CellType2"]=='CD4 TCM'])
    thelperDF = thelperDF.append(CITE_DF.loc[CITE_DF["CellType2"]=='CD4 TEM'])
    thelperDF = thelperDF.sample(100)
    
    cd8DF = CITE_DF.loc[CITE_DF["CellType2"]=='CD8 Naive']
    cd8DF = cd8DF.append(CITE_DF.loc[CITE_DF["CellType2"]=='CD8 TCM'])
    cd8DF = cd8DF.append(CITE_DF.loc[CITE_DF["CellType2"]=='CD8 TEM'])
    cd8DF = cd8DF.sample(100)

    
    treg_abundances = []
    thelper_abundances = []
    nk_abundances = []
    cd8_abundances = []
    for e in epitopesDF.Epitope:

        if e == 'CD25':
            convFact = 77.136987
        elif e == 'CD122':
            convFact = 332.680090
        elif e == "CD127":
            convFact = 594.379215
        else:
            convFact = meanConv

        
        citeVal = tregDF[e].to_numpy() 
        abundance = citeVal*convFact
        treg_abundances.append(abundance)

        citeVal = thelperDF[e].to_numpy() 
        abundance = citeVal*convFact
        thelper_abundances.append(abundance)

        citeVal = nkDF[e].to_numpy() 
        abundance = citeVal*convFact
        nk_abundances.append(abundance)

        citeVal = cd8DF[e].to_numpy()  
        abundance = citeVal*convFact
        cd8_abundances.append(abundance)
    
    # Import actual abundance into dadaframe by multiplying by average abundance (CD25, CD122, CD132 exceptions)
    epitopesDF['Treg'] = treg_abundances
    epitopesDF['Thelper'] = thelper_abundances
    epitopesDF['NK'] = nk_abundances
    epitopesDF['CD8'] = cd8_abundances

    epitopesDF['Selectivity'] = -1
    #print(epitopesDF)
    
    targCell = 'Treg'
    offTCells = ['Thelper','CD8','NK']

    # Feed actual abundance into modeling

    #print(epitopesDF)

    abundanceDF = pd.DataFrame(columns={"Receptor", "Cell Type", "Abundance"})

    receptors = ['CD25','CD122']
    cells = [targCell] + offTCells

    for r in receptors:
        for cell in cells:
            abun = epitopesDF.loc[epitopesDF['Epitope']==r][cell].sample().item().mean()
            abundanceDF = abundanceDF.append(pd.DataFrame({"Receptor": [r], "Cell Type": cell, "Abundance": abun}))

    abundanceDF.to_csv(join(path_here, "data/CiteAbundance.csv"), index=False)
    
    print("stop")



    for e in epitopesDF['Epitope'].unique():
        print(e)
        selectedDF = epitopesDF.loc[(epitopesDF.Epitope == e)]
        sampleDF = selectedDF.sample()
        optSelectivity = optimizeDesign(ax[1], targCell, offTCells, sampleDF)
        print(optSelectivity)
        epitopesDF.loc[epitopesDF['Epitope'] == e, 'Selectivity'] = optSelectivity

    

    epitopesDF = epitopesDF.drop(['Treg','Thelper','NK','CD8'],axis=1)
    epitopesDF.to_csv(join(path_here, "data/epitopeSelectivityList.csv"), index=False)

    return f

def cytBindingModel_bispecOpt(df, recXaff, cellType, x=False):
    """Runs binding model for a given mutein, valency, dose, and cell type."""

    mut = 'IL2'
    val = 1
    doseVec = np.array([0.1])
    date = '3/15/2019'

    recXaff = np.power(10,recXaff)

    #print(df)
    #print(cellType)
    recX = df
    

    recDF =  pd.read_csv(join(path_here, "data/CiteAbundance.csv"))

    recCount = np.ravel([recDF.loc[(recDF.Receptor == "CD25") & (recDF["Cell Type"] == cellType)]["Abundance"].item(),
                         recDF.loc[(recDF.Receptor == "CD122") & (recDF["Cell Type"] == cellType)]["Abundance"].item(), recX])


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
    if date:
        convDict = getBindDict()
        if cellType[-1] == "$":  # if it is a binned pop, use ave fit
            output *= convDict.loc[(convDict.Date == date) & (convDict.Cell == cellType[0:-13])].Scale.values
        else:
            output *= convDict.loc[(convDict.Date == date) & (convDict.Cell == cellType)].Scale.values
    return output



def minSelecFunc(x, df, targCell, offTCells):
    """Provides the function to be minimized to get optimal selectivity"""
    targetBound = 0
    offTargetBound = 0

    recXaff = x

    for count in df[targCell].item():
        targetBound += cytBindingModel_bispecOpt(count, recXaff,targCell)
    for cellT in offTCells:
        for count in df[cellT].item():
            offTargetBound += cytBindingModel_bispecOpt(count, recXaff, cellT)
    
    return (offTargetBound) / (targetBound)

def optimizeDesign(ax, targCell, offTcells, epitopeDF, legend=True):
    """ A more general purpose optimizer """
    vals = np.arange(1.01, 10, step=0.15)
    sigDF = pd.DataFrame()



    optDF = pd.DataFrame(columns={"Valency", "Selectivity", "IL2Rα", r"IL-2Rβ/γ$_c$"})

    if targCell == "NK":
        X0 = [6.0, 8] 
    else:
        X0 = [7.0]  

    optBnds = Bounds(np.full_like(X0, 6.0), np.full_like(X0, 9.0))

    optimized = minimize(minSelecFunc, X0, bounds=optBnds, args=(epitopeDF, targCell, offTcells), jac="3-point")
    optSelectivity = optimized.fun[0]
    #print(optSelectivity)

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
    #print(baselineAcc)
    #print(np.where((factors == "CD25")))
    for marker in factors:
        SVMmod = SVC()
        #print(marker)
        markerCol = X[:, np.where(factors == marker)]
        CD25MarkX = np.hstack((CD25col, markerCol.reshape(-1, 1)))
        markAcc = SVMmod.fit(CD25MarkX, TregY).score(CD25MarkX, TregY)
        #print(markAcc)
        AccDF = AccDF.append(pd.DataFrame({"Markers": [marker], "Accuracy": [markAcc]}))

    AccDF = AccDF.sort_values(by="Accuracy")
    markers = copy(AccDF.tail(numFactors).Markers.values) #Here
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