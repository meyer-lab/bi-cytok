"""
Unit test file.
"""

import numpy as np
import pandas as pd

from bicytok.selectivityFuncs import calcReceptorAbundances, optimizeSelectivityAffs, get_affs
from bicytok.MBmodel import cytBindingModel


def test_optimize_design():
    targCell = "Treg"
    offTCells = ["CD8 Naive", "NK", "CD8 TEM", "CD4 Naive", "CD4 CTL"]
    cells = offTCells + [targCell]

    epitopesList = pd.read_csv("./bicytok/data/epitopeList.csv")
    epitopes = list(epitopesList["Epitope"].unique())
    epitopesDF = calcReceptorAbundances(epitopes, cells)

    optimizeSelectivityAffs(
        signal="CD122",
        targets=["CD25"],
        targCell=targCell,
        offTCells=offTCells,
        selectedDF=epitopesDF,
        dose=0.1,
        valencies=np.array([[2, 2]]),
        prevOptAffs=[8.0, 8.0],
    )


def test_binding_model():
    assert cytBindingModel(recCount=np.array([4000., 3400.]), recXaffs=get_affs(np.array([8., 8.])),
        dose=0.1, vals=np.array([[1, 1]])) == 4.070165414304938e-5
    assert cytBindingModel(recCount=np.array([6000., 2100.]), recXaffs=get_affs(np.array([7.6, 8.2])),
        dose=1., vals=np.array([[4, 4]])) == 0.0009870173680610606
    assert cytBindingModel(recCount=np.array([4000., 3400., 5700., 33800.]),
        recXaffs=get_affs(np.array([8.9, 7.0, 8.0, 8.0])), dose=0.1,
        vals=np.array([[1, 4, 4, 4]])) == 0.017104443169046135