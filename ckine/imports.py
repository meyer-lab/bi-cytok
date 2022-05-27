"""File that deals with everything about importing and sampling."""
import os
from functools import lru_cache
from os.path import join
import numpy as np
import pandas as pd

path_here = os.path.dirname(os.path.dirname(__file__))


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
def makeCITEdf():
    """Makes cite surface epitope csv for given cell type, DON'T USE THIS UNLESS DATA NEEDS RESTRUCTURING"""
    """
    matrixDF = pd.read_csv(join(path_here, "ckine/data/CITEmatrix.gz"), compression='gzip', header=0, sep=' ', quotechar='"', error_bad_lines=False)
    matrixDF = matrixDF.iloc[:, 0:-2]
    matrixDF.columns = ["Marker", "Cell", "Number"]
    matrixDF.to_csv(join(path_here, "ckine/data/CITEmatrix.csv"), index=False)
    """
    featureDF = pd.read_csv(join(path_here, "ckine/data/CITEfeatures.csv"))
    matrixDF = pd.read_csv(join(path_here, "ckine/data/CITEmatrix.csv")).iloc[1::, :]
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
    return matrixDF  # , featureDF, metaDF


def importCITE():
    """Downloads all surface markers and cell types"""
    CITEmarkerDF = pd.read_csv(join(path_here, "ckine/data/CITEdata_SurfMarkers.zip"))
    return CITEmarkerDF
