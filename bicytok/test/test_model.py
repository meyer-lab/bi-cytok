"""
Unit test file.
"""

import numpy as np
import pandas as pd
from bicytok.selectivityFuncs import getSampleAbundances, optimizeDesign
from bicytok.distanceMetricFuncs import KL_EMD_1D, KL_EMD_2D, KL_EMD_3D


def test_optimize_design():
    targCell = "Treg"
    offTCells = ["CD8 Naive", "NK", "CD8 TEM", "CD4 Naive", "CD4 CTL"]
    cells = offTCells + [targCell]

    epitopesList = pd.read_csv("./bicytok/data/epitopeList.csv")
    epitopes = list(epitopesList["Epitope"].unique())
    epitopesDF = getSampleAbundances(epitopes, cells)

    optimizeDesign(
        signal="CD122",
        targets=["CD25"],
        targCell=targCell,
        offTCells=offTCells,
        selectedDF=epitopesDF,
        dose=0.1,
        valencies=np.array([[2, 2]]),
        prevOptAffs=[8.0, 8.0],
    )

def sample_data():
    np.random.seed(0)
    recAbundances = np.random.rand(100, 10) * 10
    targ = np.random.choice([True, False], size=100, p=[0.3, 0.7])
    offTarg = ~targ
    return recAbundances, targ, offTarg

def test_KL_EMD_1D():
    recAbundances, targ, offTarg = sample_data()
    targ = np.array(targ, dtype=bool)
    offTarg = np.array(offTarg, dtype=bool)
    
    KL_div_vals, EMD_vals = KL_EMD_1D(recAbundances, targ, offTarg)

    assert len(KL_div_vals) == recAbundances.shape[1]
    assert len(EMD_vals) == recAbundances.shape[1]
    print([type(i) for i in np.append(targ, offTarg)])

def test_KL_EMD_2D():
    recAbundances, targ, offTarg = sample_data()
    targ = np.array(targ, dtype=bool)
    offTarg = np.array(offTarg, dtype=bool)
    KL_div_vals, EMD_vals = KL_EMD_2D(recAbundances, targ, offTarg)

    assert KL_div_vals.shape == (recAbundances.shape[1], recAbundances.shape[1])
    assert EMD_vals.shape == (recAbundances.shape[1], recAbundances.shape[1])
    assert np.all(np.isnan(KL_div_vals) | (KL_div_vals >= 0))
    assert np.all(np.isnan(EMD_vals) | (EMD_vals >= 0))

def test_invalid_inputs():
    recAbundances = np.random.rand(100, 10)
    targ = np.random.choice([True, False], size=100, p=[0.3, 0.7])
    offTarg = ~targ

    # Test invalid inputs for KL_EMD_1D
    try:
        KL_EMD_1D(recAbundances, np.arange(100), offTarg)
    except AssertionError:
        print("Caught expected error for non-boolean targ/offTarg in KL_EMD_1D.")

    try:
        KL_EMD_1D(recAbundances, np.zeros(100, dtype=bool), offTarg)
    except AssertionError:
        print("Caught expected error for no target cells in KL_EMD_1D.")

    try:
        KL_EMD_1D(recAbundances, targ, np.zeros(100, dtype=bool))
    except AssertionError:
        print("Caught expected error for no off-target cells in KL_EMD_1D.")

    # Test invalid inputs for KL_EMD_2D
    try:
        KL_EMD_2D(recAbundances, np.arange(100), offTarg)
    except AssertionError:
        print("Caught expected error for non-boolean targ/offTarg in KL_EMD_2D.")

    try:
        KL_EMD_2D(recAbundances, np.zeros(100, dtype=bool), offTarg)
    except AssertionError:
        print("Caught expected error for no target cells in KL_EMD_2D.")

    try:
        KL_EMD_2D(recAbundances, targ, np.zeros(100, dtype=bool))
    except AssertionError:
        print("Caught expected error for no off-target cells in KL_EMD_2D.")

if __name__ == "__main__":
    test_KL_EMD_1D()
    test_KL_EMD_2D()
    test_invalid_inputs()