"""File that deals with everything about importing and sampling."""
import os
from functools import lru_cache
from os.path import join
import numpy as np
import pandas as pd

path_here = os.path.dirname(os.path.dirname(__file__))


@lru_cache(maxsize=None)
def import_pstat(combine_samples=True):
    """ Loads CSV file containing pSTAT5 levels from Visterra data. Incorporates only Replicate 1 since data missing in Replicate 2. """
    path = os.path.dirname(os.path.dirname(__file__))
    data = np.array(pd.read_csv(join(path, "ckine/data/pSTAT_data.csv"), encoding="latin1"))
    ckineConc = data[4, 2:14]
    tps = np.array([0.5, 1.0, 2.0, 4.0]) * 60.0
    # 4 time points, 10 cell types, 12 concentrations, 2 replicates
    IL2_data = np.zeros((40, 12))
    IL2_data2 = IL2_data.copy()
    IL15_data = IL2_data.copy()
    IL15_data2 = IL2_data.copy()
    cell_names = list()
    for i in range(10):
        cell_names.append(data[12 * i + 3, 1])
        # Subtract the zero treatment plates before assigning to returned arrays
        if i <= 4:
            zero_treatment = data[12 * (i + 1), 13]
            zero_treatment2 = data[8 + (12 * i), 30]
        else:
            zero_treatment = data[8 + (12 * i), 13]
            zero_treatment2 = data[8 + (12 * i), 30]
        # order of increasing time by cell type
        IL2_data[4 * i: 4 * (i + 1), :] = np.flip(data[6 + (12 * i): 10 + (12 * i), 2:14].astype(float) - zero_treatment, 0)
        IL2_data2[4 * i: 4 * (i + 1), :] = np.flip(data[6 + (12 * i): 10 + (12 * i), 19:31].astype(float) - zero_treatment2, 0)
        IL15_data[4 * i: 4 * (i + 1), :] = np.flip(data[10 + (12 * i): 14 + (12 * i), 2:14].astype(float) - zero_treatment, 0)
        IL15_data2[4 * i: 4 * (i + 1), :] = np.flip(data[10 + (12 * i): 14 + (12 * i), 19:31].astype(float) - zero_treatment2, 0)

    if combine_samples is False:
        return ckineConc, cell_names, IL2_data, IL2_data2, IL15_data, IL15_data2

    for i in range(IL2_data.shape[0]):
        for j in range(IL2_data.shape[1]):
            # take average of both replicates if specific entry isn't nan
            IL2_data[i, j] = np.nanmean(np.array([IL2_data[i, j], IL2_data2[i, j]]))
            IL15_data[i, j] = np.nanmean(np.array([IL15_data[i, j], IL15_data2[i, j]]))

    dataMean = pd.DataFrame(
        {
            "Cells": np.tile(np.repeat(cell_names, 48), 2),
            "Ligand": np.concatenate((np.tile(np.array("IL2"), 480), np.tile(np.array("IL15"), 480))),
            "Time": np.tile(np.repeat(tps, 12), 20),
            "Concentration": np.tile(ckineConc, 80),
            "RFU": np.concatenate((IL2_data.reshape(480), IL15_data.reshape(480))),
        }
    )

    return ckineConc, cell_names, IL2_data, IL15_data, dataMean


# Receptor Quant - Beads (4/23 & 4/26)


channels = {}
channels["A"] = ["VL1-H", "BL5-H", "RL1-H", "RL1-H", "RL1-H", "Width"]
channels["C"] = ["VL4-H", "VL6-H", "BL1-H", "BL3-H"]
channels["D"] = ["VL1-H", "VL1-H", "VL1-H", "VL1-H", "VL1-H"]
channels["E"] = ["VL6-H", "BL3-H", "BL5-H", "BL5-H", "BL5-H", "BL5-H", "BL5-H"]
channels["F"] = channels["G"] = channels["H"] = ["RL1-H", "RL1-H", "RL1-H", "RL1-H", "RL1-H"]
channels["I"] = ["BL1-H", "BL1-H", "BL1-H", "BL1-H", "BL1-H"]

receptors = {}
receptors["A"] = ["CD25", "CD122", "CD132", "IL15(1)", "IL15(2)", " "]
receptors["C"] = ["CD3", "CD4", "CD127", "CD45RA"]
receptors["D"] = ["CD25", "CD25", "CD25", "CD25", "CD25"]
receptors["E"] = ["CD8", "CD56", "CD122", "CD122", "CD122", "CD122", "CD122"]
receptors["F"] = ["CD132", "CD132", "CD132", "CD132", "CD132"]
receptors["G"] = ["IL15(1)", "IL15(1)", "IL15(1)", "IL15(1)", "IL15(1)"]
receptors["H"] = ["IL15(2)", "IL15(2)", "IL15(2)", "IL15(2)", "IL15(2)"]
receptors["I"] = ["CD127", "CD127", "CD127", "CD127", "CD127"]


@lru_cache(maxsize=None)
def import_pstat_all(singleCell=False):
    """ Loads CSV file containing all WT and Mutein pSTAT responses and moments"""
    WTbivDF = pd.read_csv(join(path_here, "ckine/data/WTDimericMutSingleCellData.csv"), encoding="latin1")
    monDF = pd.read_csv(join(path_here, "ckine/data/MonomericMutSingleCellData.csv"), encoding="latin1")
    respDF = pd.concat([WTbivDF, monDF])
    if singleCell:
        WTbivDFbin = pd.read_csv(join(path_here, "ckine/data/WTDimericMutSingleCellDataBin.csv"), encoding="latin1")
        monDFbin = pd.read_csv(join(path_here, "ckine/data/MonomericMutSingleCellDataBin.csv"), encoding="latin1")
        respDFbin = pd.concat([WTbivDFbin, monDFbin])
        respDFbin = respDFbin.loc[respDFbin["Bin"].isin([1, 3])]
        respDFbin.loc[respDFbin["Bin"] == 1, "Cell"] += r" $IL2Ra^{lo}$"
        respDFbin.loc[respDFbin["Bin"] == 3, "Cell"] += r" $IL2Ra^{hi}$"
        respDF = pd.concat([respDF, respDFbin])

    respDF.loc[(respDF.Bivalent == 0), "Ligand"] = (respDF.loc[(respDF.Bivalent == 0)].Ligand + " (Mono)").values
    respDF.loc[(respDF.Bivalent == 1), "Ligand"] = (respDF.loc[(respDF.Bivalent == 1)].Ligand + " (Biv)").values

    return respDF


@lru_cache(maxsize=None)
def getBindDict():
    """Gets binding to pSTAT fluorescent conversion dictionary"""
    path = os.path.dirname(os.path.dirname(__file__))
    bindingDF = pd.read_csv(join(path, "ckine/data/BindingConvDict.csv"), encoding="latin1")
    return bindingDF


@lru_cache(maxsize=None)
def importReceptors():
    """Makes Complete receptor expression Dict"""
    recDF = pd.read_csv(join(path_here, "ckine/data/RecQuantitation.csv"))
    recDFbin = pd.read_csv(join(path_here, "ckine/data/BinnedReceptorData.csv"))
    recDFbin = recDFbin.loc[recDFbin["Bin"].isin([1, 3])]
    recDFbin.loc[recDFbin["Bin"] == 1, "Cell Type"] += r" $IL2Ra^{lo}$"
    recDFbin.loc[recDFbin["Bin"] == 3, "Cell Type"] += r" $IL2Ra^{hi}$"
    recDF = pd.concat([recDF, recDFbin])
    return recDF


@lru_cache(maxsize=None)

def importCITE():
    """Makes cite surface epitope csv for given cell type, DON'T USE THIS UNLESS DATA NEEDS RESTRUCTURING"""
    """
    matrixDF = pd.read_csv(join(path_here, "ckine/data/CITEmatrix.gz"), compression='gzip', header=0, sep=' ', quotechar='"', error_bad_lines=False)
    matrixDF = matrixDF.iloc[:, 0:-2]
    matrixDF.columns = ["Marker", "Cell", "Number"]
    matrixDF.to_csv(join(path_here, "ckine/data/CITEmatrix.csv"), index=False)
    """
    featureDF = pd.read_csv(join(path_here, "ckine/data/CITEfeatures.csv"))
    matrixDF = pd.read_csv(join(path_here, "ckine/data/CITEmatrix.csv")).iloc[1:: , :]
    metaDF = pd.read_csv(join(path_here, "ckine/data/metaData3P.csv"))

    metaDF['cellNumber'] = metaDF.index + 1
    cellNums = metaDF.cellNumber.values
    cellT1 = metaDF["celltype.l1"].values
    cellT2 = metaDF["celltype.l2"].values
    cellT3 = metaDF["celltype.l3"].values
    cellTDict1 = {cellNums[i]: cellT1[i] for i in range(len(cellNums))}
    cellTDict2 = {cellNums[i]: cellT2[i] for i in range(len(cellNums))}
    cellTDict3 = {cellNums[i]: cellT3[i] for i in range(len(cellNums))}

    featureDF['featNumber'] = featureDF.index + 1
    featNums = featureDF.featNumber.values
    features = featureDF.Marker.values
    featDict = {featNums[i]: features[i] for i in range(len(featNums))}
    matrixDF["Marker"] = matrixDF["Marker"].replace(featDict)

    categories1 = metaDF["celltype.l1"].unique()
    categories2 = metaDF["celltype.l2"].unique()
    categories3 = metaDF["celltype.l3"].unique()

    matrixDF = matrixDF.pivot(index=["Cell"], columns="Marker", values="Number").reset_index().fillna(0)

    matrixDF["CellType1"] = pd.Categorical(matrixDF["Cell"].replace(cellTDict1), categories=categories1)
    matrixDF["CellType2"] = pd.Categorical(matrixDF["Cell"].replace(cellTDict2), categories=categories2)
    matrixDF["CellType3"] = pd.Categorical(matrixDF["Cell"].replace(cellTDict3), categories=categories3)
    matrixDF.to_csv(join(path_here, "ckine/data/CITEdata.csv"), index=False)
    return matrixDF #, featureDF, metaDF
