"""
Unit test file for distribution metric functions.
"""

import numpy as np
import pytest

from ..distance_metric_funcs import KL_EMD_1D, KL_EMD_2D, KL_EMD_3D
from ..imports import sample_test_data as sample_data


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


def test_KL_EMD_3D():
    recAbundances, targ, offTarg = sample_data(n_var=3)
    targ = np.array(targ, dtype=bool)
    offTarg = np.array(offTarg, dtype=bool)

    KL_div_vals, EMD_vals = KL_EMD_3D(recAbundances, targ, offTarg)

    assert KL_div_vals.shape == (
        recAbundances.shape[1],
        recAbundances.shape[1],
        recAbundances.shape[1],
    )
    assert EMD_vals.shape == (
        recAbundances.shape[1],
        recAbundances.shape[1],
        recAbundances.shape[1],
    )
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

    # Test invalid inputs for KL_EMD_3D
    with pytest.raises(AssertionError):
        KL_EMD_3D(recAbundances, np.arange(100), offTarg)

    with pytest.raises(AssertionError):
        KL_EMD_3D(recAbundances, np.full_like(targ, False), offTarg)

    with pytest.raises(AssertionError):
        KL_EMD_3D(recAbundances, targ, np.full_like(offTarg, False))
