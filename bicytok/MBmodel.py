"""
Implementation of a simple multivalent binding model.
"""

import numpy as np
import pandas as pd
from .BindingMod import polyc
from .imports import getBindDict, importReceptors


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

    mutAffDF = pd.read_csv("./bicytok/data/WTmutAffData.csv")
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
            output[i] = polyc(dose / 1e9, np.power(10, x[0]), recCount, [[val, val]], Affs)[0][1]
        else:
            output[i] = polyc(dose / 1e9, getKxStar(), recCount, [[val, val]], Affs)[0][1]  # IL2RB binding only
    if date:
        convDict = getBindDict()
        if cellType[-1] == "$":  # if it is a binned pop, use ave fit
            output *= convDict.loc[(convDict.Date == date) & (convDict.Cell == cellType[0:-13])].Scale.values
        else:
            output *= convDict.loc[(convDict.Date == date) & (convDict.Cell == cellType)].Scale.values
    return output


def cytBindingModel_basicSelec(counts) -> float:
    """Runs binding model for a given dataframe of epitope abundances"""
    mut = 'IL2'
    val = 1
    dose = 0.1

    recCount = np.ravel(counts)
    mutAffDF = pd.read_csv("./bicytok/data/WTmutAffData.csv")
    Affs = mutAffDF.loc[(mutAffDF.Mutein == mut)]
    Affs = np.power(np.array([Affs["IL2RaKD"].values, Affs["IL2RBGKD"].values]) / 1e9, -1)
    Affs = np.reshape(Affs, (1, -1))
    Affs = np.repeat(Affs, 2, axis=0)
    np.fill_diagonal(Affs, 1e2)  # Each cytokine can only bind one a and one b

    return polyc(dose / 1e9, getKxStar(), recCount, [[val, val]], Affs)[0][1]  # IL2RB binding only


# CITEseq Tetra valent exploration functions below

def cytBindingModel_CITEseq(mutAffDF, counts, betaAffs, val) -> float:
    """Runs binding model for a given epitopes abundance, betaAffinity, valency, and mutein type."""

    dose = 0.1
    recCount = np.ravel(counts)

    Affs = np.power(np.array([mutAffDF["IL2RaKD"].values, [betaAffs]]) / 1e9, -1)

    Affs = np.reshape(Affs, (1, -1))
    Affs = np.repeat(Affs, 2, axis=0)
    np.fill_diagonal(Affs, 1e2)  # Each cytokine can only bind one a and one b
    vals = np.full((1, 2), val)

    return polyc(dose / 1e9, getKxStar(), recCount, vals, Affs)[0][1]


def cytBindingModel_bispecCITEseq(counts, betaAffs, recXaff, vals, mut, x=False) -> float:
    """Runs bispecific binding model built for CITEseq data for a given mutein, epitope, valency, dose, and cell type."""

    recXaff = np.power(10, recXaff)
    doseVec = np.array([0.1])
    recCount = np.ravel(counts)
    mutAffDF = pd.read_csv(join(path_here, "bicytok/data/WTmutAffData.csv"))
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
            output[i] = polyc(dose / (val * 1e9), np.power(10, x[0]), recCount, vals, Affs)[0][1]
        else:
            output[i] = polyc(dose / (val * 1e9), getKxStar(), recCount, vals, Affs)[0][1]

    return output

def cytBindingModel_bispecOpt(recCount: np.ndarray, holder: np.ndarray, dose: float, vals: np.ndarray, x=False):
    """Runs binding model for a given mutein, valency, dose, and cell type."""
    # Check that values are in correct placement, can invert
    doseVec = np.array(dose)
    Kx = getKxStar()

    if doseVec.size == 1:
        doseVec = np.array([doseVec])
    output = np.zeros(doseVec.size)

    for i, dose in enumerate(doseVec):
        output[i] = polyc(dose / (vals[0] * 1e9), Kx, recCount, [vals], holder)[0][1]

    return output
