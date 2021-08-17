"""
This creates Figure P1, dose response of all IL-2 cytokines at varing valencies using binding model.
"""

import numpy as np
# from .figureCommon import getSetup, plotDoseResponses_tetra
from MBmodel import runFullModel_bispec

def testModel():
    """Get a list of the axis objects and create a figure"""
    print("okay")
    modelDF = runFullModel_bispec(time=[0.5, 1], saveDict=False)

    print(modelDF)

    ligands = modelDF.Ligand.unique()
    cells = ["Treg", "Thelper", "NK", "CD8"]
    # ax, f = getSetup((10, 16), (len(ligands), 4))

    # for i, lig in enumerate(ligands):
    #     for j, cell in enumerate(cells):
    #         plotDoseResponses_tetra(ax[4 * i + j], modelDF, lig, cell)

    # return f
    return

#calc at different valencies
testModel()

