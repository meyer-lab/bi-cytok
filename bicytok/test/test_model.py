"""
Unit test file.
"""

import numpy as np
import pandas as pd

from bicytok.MBmodel import cytBindingModel
from bicytok.selectivityFuncs import (
    calcReceptorAbundances,
    get_affs,
    optimizeSelectivityAffs,
)


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
    assert np.isclose(
        cytBindingModel(
            recCount=np.array([4000.0, 3400.0]),
            recXaffs=get_affs(np.array([8.0, 8.0])),
            dose=0.1,
            vals=np.array([[1, 1]]),
        ),
        4.070165414304938e-5,
    )
    assert np.isclose(
        cytBindingModel(
            recCount=np.array([6000.0, 2100.0]),
            recXaffs=get_affs(np.array([7.6, 8.2])),
            dose=1.0,
            vals=np.array([[4, 4]]),
        ),
        0.0009870173680610606,
    )
    assert np.isclose(
        cytBindingModel(
            recCount=np.array([4000.0, 3400.0, 5700.0, 33800.0]),
            recXaffs=get_affs(np.array([8.9, 7.0, 8.0, 8.0])),
            dose=0.1,
            vals=np.array([[1, 4, 4, 4]]),
        ),
        0.017104443169046135,
    )
