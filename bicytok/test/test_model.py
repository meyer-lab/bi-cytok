"""
Unit test file.
"""

import numpy as np
import pytest

from ..distanceMetricFuncs import KL_EMD_1D, KL_EMD_2D, KL_EMD_3D


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


def test_invalid_inputs():
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


if __name__ == "__main__":
    test_KL_EMD_1D()
    test_KL_EMD_2D()
    test_invalid_inputs()
