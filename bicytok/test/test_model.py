"""
Unit test file.
"""

import numpy as np
import pandas as pd
import pytest

from ..distanceMetricFuncs import (
    KL_EMD_1D, 
    KL_EMD_2D, 
    KL_EMD_3D
)
from ..selectivityFuncs import (
    sampleReceptorAbundances, 
    restructureAffs,
    minOffTargSelec
)
from ..MBmodel import cytBindingModel

"""
def test_optimizeSelectivityAffs():
    targCell = "Treg"
    offTCells = ["CD8 Naive", "NK", "CD8 TEM", "CD4 Naive", "CD4 CTL"]
    cells = offTCells + [targCell]

    epitopesList = pd.read_csv("./bicytok/data/epitopeList.csv")
    epitopes = list(epitopesList["Epitope"].unique())
    epitopesDF = getReceptorAbundances(epitopes, cells)

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
"""


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
    assert all([isinstance(i, np.bool) for i in np.append(targ, offTarg)])


def test_KL_EMD_2D():
    recAbundances, targ, offTarg = sample_data()
    targ = np.array(targ, dtype=bool)
    offTarg = np.array(offTarg, dtype=bool)
    KL_div_vals, EMD_vals = KL_EMD_2D(recAbundances, targ, offTarg)

    assert KL_div_vals.shape == (recAbundances.shape[1], recAbundances.shape[1])
    assert EMD_vals.shape == (recAbundances.shape[1], recAbundances.shape[1])
    assert np.all(np.isnan(KL_div_vals) | (KL_div_vals >= 0))
    assert np.all(np.isnan(EMD_vals) | (EMD_vals >= 0))


def test_invalid_distance_function_inputs():
    recAbundances = np.random.rand(100, 10)
    targ = np.random.choice([True, False], size=100, p=[0.3, 0.7])
    offTarg = ~targ

    # Test invalid inputs for KL_EMD_1D
    with pytest.raises(AssertionError):
        KL_EMD_1D(recAbundances, np.arange(100), offTarg) # non-boolean targ/offTarg

    with pytest.raises(AssertionError):
        KL_EMD_1D(recAbundances, np.zeros(100, dtype=bool), offTarg) # no target cells

    with pytest.raises(AssertionError):
        KL_EMD_1D(recAbundances, targ, np.zeros(100, dtype=bool)) # no off-target cells

    # Test invalid inputs for KL_EMD_2D
    with pytest.raises(AssertionError):
        KL_EMD_2D(recAbundances, np.arange(100), offTarg) # non-boolean targ/offTarg

    with pytest.raises(AssertionError):
        KL_EMD_2D(recAbundances, np.zeros(100, dtype=bool), offTarg) # no target cells

    with pytest.raises(AssertionError):
        KL_EMD_2D(recAbundances, targ, np.zeros(100, dtype=bool)) # no off-target cells


def test_invalid_model_function_inputs():
    # Test invalid inputs for restructuringAffs
    with pytest.raises(AssertionError):
        restructureAffs(np.array([[8.0, 8.0], [8.0, 8.0]])) # 2D receptor affinities

    with pytest.raises(AssertionError):
        restructureAffs(np.array([])) # empty array

    # Assign default values for cytBindingModel and minOffTargSelec
    dose = 0.1
    recCounts1D = np.random.rand(3) # for testing one cell
    recCounts2D = np.random.rand(100, 3) # for testing multiple cells
    valencies = np.array([[1, 1, 1]])
    monomerAffs = np.array([8.0, 8.0, 8.0])
    modelAffs = restructureAffs(monomerAffs)

    # Test invalid inputs for cytBindingModel
    with pytest.raises(AssertionError):
        cytBindingModel(dose, np.random.rand(100, 3, 3), valencies, modelAffs) # 3D receptor counts

    with pytest.raises(AssertionError):
        cytBindingModel(dose, recCounts2D, np.array([[1, 1, 1, 1]]), modelAffs) # wrong number of valenciess

    with pytest.raises(AssertionError):
        cytBindingModel(dose, recCounts2D, valencies, restructureAffs(np.array([8.0, 8.0, 8.0, 8.0]))) # wrong number of complexes

    with pytest.raises(AssertionError):
        cytBindingModel(dose, recCounts1D, np.array([[1, 1]]), restructureAffs(np.array([8.0, 8.0]))) # 1D mismatched number of types of receptors

    with pytest.raises(AssertionError):
        cytBindingModel(dose, recCounts2D, np.array([[1, 1]]), restructureAffs(np.array([8.0, 8.0]))) # 2D mismatched number of types of receptors

    # Test invalid inputs for minOffTargSelec
    with pytest.raises(AssertionError):
        minOffTargSelec(modelAffs, recCounts2D, np.random.rand(100, 4), dose, valencies) # mismatched number of types of receptors

    # Assign default values for sampleReceptorAbundances
    df = pd.DataFrame({
        "Cell Type": ["Treg"] * 100,
        "CD122": np.random.rand(100),
        "CD25": np.random.rand(100),
    })
    numCells = 50

    # Test invalid inputs for sampleReceptorAbundances
    with pytest.raises(AssertionError):
        sampleReceptorAbundances(df, 200) # requested sample size greater than available cells

    with pytest.raises(AssertionError):
        sampleReceptorAbundances(df.drop(columns="Cell Type"), numCells) # lack of cell type column


if __name__ == "__main__":
    test_KL_EMD_1D()
    test_KL_EMD_2D()
    test_invalid_distance_function_inputs()
    test_invalid_model_function_inputs()