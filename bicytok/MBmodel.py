"""
Implementation of a simple multivalent binding model.
"""

from os.path import dirname, join
import numpy as np
import pandas as pd
from .BindingMod import polyc
from .imports import getBindDict, importReceptors

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
            output[i] = polyc(dose / 1e9, np.power(10, x[0]), recCount, [[val, val]], [1.0], Affs)[1][0][1]
        else:
            output[i] = polyc(dose / 1e9, getKxStar(), recCount, [[val, val]], [1.0], Affs)[1][0][1]  # IL2RB binding only
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
    mutAffDF = pd.read_csv(join(path_here, "bicytok/data/WTmutAffData.csv"))
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
            output[i] = polyc(dose / 1e9, np.power(10, x[0]), recCount, [[val, val]], [1.0], Affs)[1][0][1]
        else:
            print(dose / 1e9, getKxStar(), recCount, [[val, val]], Affs)
            output[i] = polyc(dose / 1e9, getKxStar(), recCount, [[val, val]], [1.0], Affs)[1][0][1]  # IL2RB binding only
    return output


# CITEseq Tetra valent exploration functions below

def cytBindingModel_CITEseq(counts, betaAffs, val, mut, x=False, date=False):
    """Runs binding model for a given epitopes abundance, betaAffinity, valency, and mutein type."""

    doseVec = np.array([0.1])
    recCount = np.ravel(counts)

    mutAffDF = pd.read_csv(join(path_here, "bicytok/data/WTmutAffData.csv"))
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
            output[i] = polyc(dose / 1e9, np.power(10, x[0]), recCount, [[val, val]], [1.0], Affs)[1][0][1]
        else:
            output[i] = polyc(dose / 1e9, getKxStar(), recCount, [[val, val]], [1.0], Affs)[1][0][1]

    return output


def cytBindingModel_bispecCITEseq(counts, betaAffs, recXaff, val, mut, x=False):
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
            output[i] = polyc(dose / (val * 1e9), np.power(10, x[0]), recCount, [[val, val, val]], [1.0], Affs)[1][0][1]
        else:
            output[i] = polyc(dose / (val * 1e9), getKxStar(), recCount, [[val, val, val]], [1.0], Affs)[1][0][1]

    return output

def cytBindingModel_bispecOpt(secondary, epitope, IL2RaRecs, secondaryRecs, epitopeRecs, recXaff1, recXaff2, recXaff3, dose, val, x=False):
    """Runs binding model for a given mutein, valency, dose, and cell type."""
    doseVec = np.array(dose)

    affs = pd.DataFrame()
    affs = np.append(affs, np.power(10, recXaff1))
    affs = np.append(affs, np.power(10, recXaff2))

    if secondary == 'CD122':
        affs = np.append(affs, np.power(10, recXaff3))
        vals = [[val, val, val]]

        if epitope == 'CD25':
            holder = np.full((3, 2), 1e2)
            holder[0, 0] = affs[0]
            holder[1, 1] = affs[1]
            holder[2, 0] = affs[2]
            counts = [IL2RaRecs, secondaryRecs]
        else:
            holder = np.full((3, 3), 1e2)
            np.fill_diagonal(holder, affs)
            counts = [IL2RaRecs, secondaryRecs, epitopeRecs]

    else:
        vals = [[val, val]]
        holder = np.full((2, 2), 1e2)
        np.fill_diagonal(holder, affs)
        counts = [IL2RaRecs, secondaryRecs]

    affs = holder
    recCount = np.ravel(counts)

    # Check that values are in correct placement, can invert

    if doseVec.size == 1:
        doseVec = np.array([doseVec])
    output = np.zeros(doseVec.size)

    for i, dose in enumerate(doseVec):
        if x:
            output[i] = polyc(dose / (val * 1e9), np.power(10, x[0]), recCount, vals, [1.0], affs)[1][0][1]
        else:
            output[i] = polyc(dose / (val * 1e9), getKxStar(), recCount, vals, [1.0], affs)[1][0][1]
    
    return output
