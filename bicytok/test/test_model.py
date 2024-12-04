"""
Unit test file.
"""

import numpy as np
import pandas as pd
import pytest  
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

@pytest.fixture
def sample_data():
    np.random.seed(0)
    recAbundances = np.random.rand(100, 10) * 10
    targ = np.random.choice([True, False], size=100, p=[0.3, 0.7])
    offTarg = ~targ
    return recAbundances, targ, offTarg

def test_KL_EMD_1D(sample_data):
    recAbundances, targ, offTarg = sample_data

    KL_div_vals, EMD_vals = KL_EMD_1D(recAbundances, targ, offTarg)

    assert KL_div_vals.shape == (recAbundances.shape[1],)
    assert EMD_vals.shape == (recAbundances.shape[1],)
    assert np.all(np.isnan(KL_div_vals) | (KL_div_vals >= 0))
    assert np.all(np.isnan(EMD_vals) | (EMD_vals >= 0))

    controlled_recAbundances = np.array([[1, 2], [2, 3], [3, 4]])
    controlled_targ = np.array([True, False, True])
    controlled_offTarg = np.array([False, True, False])
    KL_div, EMD = KL_EMD_1D(controlled_recAbundances, controlled_targ, controlled_offTarg)
    assert not np.isnan(KL_div[0])
    assert not np.isnan(EMD[0])

def test_KL_EMD_2D(sample_data):
    recAbundances, targ, offTarg = sample_data

    KL_div_vals, EMD_vals = KL_EMD_2D(recAbundances, targ, offTarg)

    assert KL_div_vals.shape == (recAbundances.shape[1], recAbundances.shape[1])
    assert EMD_vals.shape == (recAbundances.shape[1], recAbundances.shape[1])
    assert np.all(np.isnan(KL_div_vals) | (KL_div_vals >= 0))
    assert np.all(np.isnan(EMD_vals) | (EMD_vals >= 0))


def test_invalid_inputs():
    recAbundances = np.random.rand(100, 10)
    targ = np.random.choice([True, False], size=100, p=[0.3, 0.7])
    offTarg = ~targ

    # Test non-boolean targ/offTarg
    with pytest.raises(AssertionError):
        KL_EMD_1D(recAbundances, np.arange(100), offTarg)

    # Test no target cells
    with pytest.raises(AssertionError):
        KL_EMD_1D(recAbundances, np.zeros(100, dtype=bool), offTarg)

    # Test no off-target cells
    with pytest.raises(AssertionError):
        KL_EMD_1D(recAbundances, targ, np.zeros(100, dtype=bool))

if __name__ == "__main__":
    pytest.main()
