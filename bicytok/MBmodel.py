"""
Implementation of a simple multivalent binding model.
"""

from os.path import dirname, join
import numpy as np
import pandas as pd
<<<<<<< HEAD:ckine/MBmodel.py
from scipy.optimize import root
from .imports import getBindDict, importReceptors
=======
from valentbind import polyc
from .imports import import_pstat_all, getBindDict, importReceptors
>>>>>>> main:bicytok/MBmodel.py

path_here = dirname(dirname(__file__))


def getKxStar():
    return 2.24e-12


def cytBindingModel(mut, val, doseVec, cellType, x=False, date=False):
    """Runs binding model for a given mutein, valency, dose, and cell type."""
    recDF = importReceptors()
    recCount = np.ravel(
        [
            recDF.loc[(recDF.Receptor == "IL2Ra") & (recDF["Cell Type"] == cellType)].Mean.values,
            recDF.loc[(recDF.Receptor == "IL2Rb") & (recDF["Cell Type"] == cellType)].Mean.values,
        ]
    )

    mutAffDF = pd.read_csv(join(path_here, "bicytok/data/WTmutAffData.csv"))
    Affs = mutAffDF.loc[(mutAffDF.Mutein == mut)]
    Affs = np.power(np.array([Affs["IL2RaKD"].values, Affs["IL2RBGKD"].values]) / 1e9, -1)
    Affs = np.reshape(Affs, (1, -1))
    Affs = np.repeat(Affs, 2, axis=0)
    np.fill_diagonal(Affs, 1e2)  # Each cytokine can only bind one a and one b

    print(type(doseVec))
    if doseVec.size == 1:
        doseVec = np.array([doseVec])
    output = np.zeros(doseVec.size)

    for i, dose in enumerate(doseVec):
        if x:
            output[i] = polyc(dose / 1e9, np.power(10, x[0]), recCount, [[val, val]], [1.0], Affs)[0][1]
        else:
            output[i] = polyc(dose / 1e9, getKxStar(), recCount, [[val, val]], [1.0], Affs)[0][1]  # IL2RB binding only
    if date:
        convDict = getBindDict()
        if cellType[-1] == "$":  # if it is a binned pop, use ave fit
            output *= convDict.loc[(convDict.Date == date) & (convDict.Cell == cellType[0:-13])].Scale.values
        else:
            output *= convDict.loc[(convDict.Date == date) & (convDict.Cell == cellType)].Scale.values
    return output


def cytBindingModel_basicSelec(counts, x=False, date=False):
    """Runs binding model for a given dataframe of epitope abundances"""
    mut = 'IL2'
    val = 1
    doseVec = np.array([0.1])
    #
    recCount = np.ravel(counts)
    #
    mutAffDF = pd.read_csv(join(path_here, "ckine/data/WTmutAffData.csv"))
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

<<<<<<< HEAD:ckine/MBmodel.py
    return output
=======
        entry = group.Mean.values
        if len(entry) >= 1:
            expVal = np.mean(entry)
            # print(type(conc))
            predVal = cytBindingModel(ligName, val, conc, cell, x)
            masterSTAT = masterSTAT.append(
                pd.DataFrame(
                    {
                        "Ligand": ligName,
                        "Date": date,
                        "Cell": cell,
                        "Dose": conc,
                        "Time": time,
                        "Valency": val,
                        "Experimental": expVal,
                        "Predicted": predVal,
                    }
                )
            )

    for date in dates:
        for cell in masterSTAT.Cell.unique():
            if cell[-1] == "$":  # if it is a binned pop, use ave fit
                predVecBin = masterSTAT.loc[(masterSTAT.Date == date) & (masterSTAT.Cell == cell)].Predicted.values
                slope = dateConvDF.loc[(dateConvDF.Date == date) & (dateConvDF.Cell == cell[0:-13])].Scale.values
                masterSTAT.loc[(masterSTAT.Date == date) & (masterSTAT.Cell == cell), "Predicted"] = predVecBin * slope
            else:
                expVec = masterSTAT.loc[(masterSTAT.Date == date) & (masterSTAT.Cell == cell)].Experimental.values
                predVec = masterSTAT.loc[(masterSTAT.Date == date) & (masterSTAT.Cell == cell)].Predicted.values
                slope = np.linalg.lstsq(np.reshape(predVec, (-1, 1)), np.reshape(expVec, (-1, 1)), rcond=None)[0][0]
                masterSTAT.loc[(masterSTAT.Date == date) & (masterSTAT.Cell == cell), "Predicted"] = predVec * slope
                dateConvDF = dateConvDF.append(pd.DataFrame({"Date": date, "Scale": slope, "Cell": cell}))
    if saveDict:
        dateConvDF.set_index("Date").to_csv(join(path_here, "bicytok/data/BindingConvDict.csv"))

    if x:
        return np.linalg.norm(masterSTAT.Predicted.values - masterSTAT.Experimental.values)
    else:
        return masterSTAT
>>>>>>> main:bicytok/MBmodel.py


# CITEseq Tetra valent exploration functions below

def cytBindingModel_CITEseq(counts, betaAffs, val, mut, x=False, date=False):
    """Runs binding model for a given epitopes abundance, betaAffinity, valency, and mutein type."""

    doseVec = np.array([0.1])
    recCount = np.ravel(counts)

    mutAffDF = pd.read_csv(join(path_here, "ckine/data/WTmutAffData.csv"))
    Affs = mutAffDF.loc[(mutAffDF.Mutein == mut)]

    Affs = np.power(np.array([Affs["IL2RaKD"].values, [betaAffs]]) / 1e9, -1)

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
            output[i] = polyc(dose / 1e9, getKxStar(), recCount, [[val, val]], [1.0], Affs)[0][1]

    return output


def cytBindingModel_bispecCITEseq(counts, betaAffs, recXaff, val, mut, x=False):
    """Runs bispecific binding model built for CITEseq data for a given mutein, epitope, valency, dose, and cell type."""

    recXaff = np.power(10, recXaff)
    doseVec = np.array([0.1])
    recCount = np.ravel(counts)

    mutAffDF = pd.read_csv(join(path_here, "ckine/data/WTmutAffData.csv"))
    Affs = mutAffDF.loc[(mutAffDF.Mutein == mut)]
    Affs = np.power(np.array([Affs["IL2RaKD"].values, [betaAffs]]) / 1e9, -1)
    Affs = np.reshape(Affs, (1, -1))
    Affs = np.append(Affs, recXaff)
    holder = np.full((3, 3), 1e2)
    np.fill_diagonal(holder, Affs)
    Affs = holder

    if doseVec.size == 1:
        doseVec = np.array([doseVec])
    output = np.zeros(doseVec.size)

    for i, dose in enumerate(doseVec):
        if x:
            output[i] = polyc(dose / (val * 1e9), np.power(10, x[0]), recCount, [[val, val, val]], [1.0], Affs)[0][1]
        else:
            output[i] = polyc(dose / (val * 1e9), getKxStar(), recCount, [[val, val, val]], [1.0], Affs)[0][1]

    return output


def cytBindingModel_bispecOpt(counts, recXaff, x=False):
    """Runs binding model for a given mutein, valency, dose, and cell type."""
<<<<<<< HEAD:ckine/MBmodel.py

    mut = 'IL2'
    val = 1
    doseVec = np.array([0.1])

    recXaff = np.power(10, recXaff)

    recCount = np.ravel(counts)
=======
    recDF = importReceptors()
    recCount = np.ravel(
        [
            recDF.loc[(recDF.Receptor == "IL2Ra") & (recDF["Cell Type"] == cellType)].Mean.values[0],
            recDF.loc[(recDF.Receptor == "IL2Rb") & (recDF["Cell Type"] == cellType)].Mean.values[0],
            recX,
        ]
    )
>>>>>>> main:bicytok/MBmodel.py

    mutAffDF = pd.read_csv(join(path_here, "bicytok/data/WTmutAffData.csv"))
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


def runFullModel_bispec(conc):
    """Runs model for all data points and outputs date conversion dict for binding to pSTAT. Can be used to fit Kx"""

    masterSTAT = pd.DataFrame(columns={"Ligand", "Dose", "Cell", "Abundance", "Affinity", "Predicted"})

    ligName = "IL2"
    # Dates = 3/15/2019, 3/27/2019, 4/18/2019
    date = "3/15/2019"
    cells = ["Treg", "Thelper", "CD8", "NK"]
    x = False
    recX_abundances = np.arange(100, 10200, 300)
    recX_affinities = [1e6, 1e8, 1e10]
    levels = ["Low", "Medium", "High"]

    for cell in cells:
        for l, recXaff in enumerate(recX_affinities):
            for recX in recX_abundances:
                # print(recX)
                predVal_bispec = cytBindingModel_bispec(ligName, 1, conc, cell, recX, recXaff, x, date)  # put in date
                masterSTAT = masterSTAT.append(
                    pd.DataFrame(
                        {"Ligand": ligName, "Dose": conc, "Cell": cell, "Abundance": recX, "Affinity": levels[l], "Predicted": predVal_bispec}
                    )
                )

    return masterSTAT
