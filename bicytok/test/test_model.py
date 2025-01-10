"""
Unit test file.
"""

import numpy as np
import pandas as pd
import pytest

from ..distanceMetricFuncs import KL_EMD_1D, KL_EMD_2D
from ..MBmodel import cytBindingModel
from ..selectivityFuncs import (
    minOffTargSelec,
    optimizeSelectivityAffs,
    restructureAffs,
    sampleReceptorAbundances,
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
    recAbundances, targ, offTarg = sample_data()

    # Test invalid inputs for KL_EMD_1D
    with pytest.raises(AssertionError):
        KL_EMD_1D(recAbundances, np.arange(100), offTarg)  # non-boolean targ/offTarg

    with pytest.raises(AssertionError):
        KL_EMD_1D(recAbundances, np.full_like(targ, False), offTarg)  # no target cells

    with pytest.raises(AssertionError):
        KL_EMD_1D(
            recAbundances, targ, np.full_like(offTarg, False)
        )  # no off-target cells

    # Test invalid inputs for KL_EMD_2D
    with pytest.raises(AssertionError):
        KL_EMD_2D(recAbundances, np.arange(100), offTarg)  # non-boolean targ/offTarg

    with pytest.raises(AssertionError):
        KL_EMD_2D(recAbundances, np.full_like(targ, False), offTarg)  # no target cells

    with pytest.raises(AssertionError):
        KL_EMD_2D(
            recAbundances, targ, np.full_like(offTarg, False)
        )  # no off-target cells


def test_optimizeSelectivityAffs():
    recAbundances, targ, offTarg = sample_data()
    recAbundances = recAbundances[:, 0:3]
    targRecs = recAbundances[targ]
    targRecs = targRecs[0:4, :]
    offTargRecs = recAbundances[offTarg]
    offTargRecs = offTargRecs[0:4, :]
    dose = 0.1
    valencies = np.array([[1, 1, 1]])

    optSelec, optParams = optimizeSelectivityAffs(
        targRecs=targRecs, offTargRecs=offTargRecs, dose=dose, valencies=valencies
    )

    assert isinstance(optSelec, float)
    assert optSelec >= 0
    assert optParams.shape == valencies[0].shape
    assert all(optParams >= 0)


# def test_binding_model():
#     assert np.isclose(
#         cytBindingModel(
#             dose=0.1,
#             recCounts=np.array([4000., 3400.]),
#             valencies=np.array([[1, 1]]),
#             monomerAffs=restructureAffs(np.array([8., 8.]))
#         )[0],
#         4.070165414304938e-5
#     )


def test_invalid_model_function_inputs():
    # Test invalid inputs for restructuringAffs
    with pytest.raises(AssertionError):
        restructureAffs(np.array([[8.0, 8.0], [8.0, 8.0]]))  # 2D receptor affinities

    with pytest.raises(AssertionError):
        restructureAffs(np.array([]))  # empty array

    # Assign default values for cytBindingModel and minOffTargSelec
    dose = 0.1
    recCounts1D = np.random.rand(3)  # for testing one cell, three receptors
    recCounts2D = np.random.rand(100, 3)  # for testing multiple cells, three receptors
    valencies = np.array([[1, 1, 1]])
    monomerAffs = np.array([8.0, 8.0, 8.0])
    modelAffs = restructureAffs(monomerAffs)

    # Test invalid inputs for cytBindingModel
    with pytest.raises(AssertionError):
        cytBindingModel(
            dose=dose,
            recCounts=np.random.rand(100, 3, 3),
            valencies=valencies,
            monomerAffs=modelAffs,
        )  # 3D receptor counts

    with pytest.raises(AssertionError):
        cytBindingModel(
            dose=dose,
            recCounts=recCounts2D,
            valencies=np.array([[1, 1, 1, 1]]),
            monomerAffs=modelAffs,
        )  # wrong number of valencies

    with pytest.raises(AssertionError):
        cytBindingModel(
            dose=dose,
            recCounts=recCounts2D,
            valencies=valencies,
            monomerAffs=restructureAffs(np.array([8.0, 8.0, 8.0, 8.0])),
        )  # wrong number of complexes

    with pytest.raises(AssertionError):
        cytBindingModel(
            dose=dose,
            recCounts=recCounts1D,
            valencies=np.array([[1, 1]]),
            monomerAffs=restructureAffs(np.array([8.0, 8.0])),
        )  # 1D mismatched number of receptors

    with pytest.raises(AssertionError):
        cytBindingModel(
            dose=dose,
            recCounts=recCounts2D,
            valencies=np.array([[1, 1]]),
            monomerAffs=restructureAffs(np.array([8.0, 8.0])),
        )  # 2D mismatched number of receptors

    # Test invalid inputs for minOffTargSelec
    with pytest.raises(AssertionError):
        minOffTargSelec(
            monomerAffs=modelAffs,
            targRecs=recCounts2D,
            offTargRecs=np.random.rand(100, 4),
            dose=dose,
            valencies=valencies,
        )  # mismatched number of receptors

    # Test invalid inputs for optimizeSelectivityAffs
    with pytest.raises(AssertionError):
        optimizeSelectivityAffs(
            targRecs=np.array([]),
            offTargRecs=recCounts2D,
            dose=dose,
            valencies=valencies,
        )  # empty target receptors

    # Assign default values for sampleReceptorAbundances
    df = pd.DataFrame(
        {
            "Cell Type": ["Treg"] * 100,
            "CD122": np.random.rand(100),
            "CD25": np.random.rand(100),
        }
    )
    numCells = 50

    # Test invalid inputs for sampleReceptorAbundances
    with pytest.raises(AssertionError):
        sampleReceptorAbundances(
            CITE_DF=df, numCells=200
        )  # requested sample size greater than available cells

    with pytest.raises(AssertionError):
        sampleReceptorAbundances(
            CITE_DF=df.drop(columns="Cell Type"), numCells=numCells
        )  # lack of cell type column


if __name__ == "__main__":
    test_KL_EMD_1D()
    test_KL_EMD_2D()
    test_invalid_distance_function_inputs()
    test_optimizeSelectivityAffs()
    # test_binding_model()
    test_invalid_model_function_inputs()
