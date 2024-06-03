"""
Unit test file.
"""

import numpy as np
import pandas as pd

from bicytok.selectivityFuncs import calcReceptorAbundances, optimizeSelectivityAffs


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
