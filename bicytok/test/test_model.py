"""
Unit test file.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ..binding_model_funcs import cyt_binding_model
from ..distance_metric_funcs import KL_EMD_1D, KL_EMD_2D, KL_EMD_3D
from ..imports import importCITE, sample_receptor_abundances
from ..selectivity_funcs import (
    min_off_targ_selec,
    optimize_affs,
    restructure_affs,
)

path_here = Path(__file__).parent.parent


def sample_data(n_obs=100, n_var=10):
    rng = np.random.default_rng(1)
    recAbundances = rng.uniform(size=(n_obs, n_var)) * 10
    targ_ind = rng.choice(n_obs, size=n_obs // 2, replace=False)
    targ = np.zeros(n_obs, dtype=bool)
    targ[targ_ind] = True
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


def test_optimize_affs():
    recAbundances, targ, offTarg = sample_data()
    recAbundances = recAbundances[:, 0:3]
    targRecs = recAbundances[targ]
    targRecs = targRecs[0:4, :]
    offTargRecs = recAbundances[offTarg]
    offTargRecs = offTargRecs[0:4, :]
    dose = 0.1
    valencies = np.array([[1, 1, 1]])

    optSelec, optAffs, optKx_star = optimize_affs(
        targRecs=targRecs, offTargRecs=offTargRecs, dose=dose, valencies=valencies
    )

    assert isinstance(optSelec, float)
    assert optSelec >= 0
    assert optAffs.shape == valencies[0].shape
    assert all(optAffs >= 0)
    assert isinstance(optKx_star, float)
    assert optKx_star > 0


def test_binding_model():
    rng = np.random.default_rng(1)

    num_receptors = 3
    num_cells = 1000

    epitopes_list = pd.read_csv(path_here / "data" / "epitopeList.csv")
    epitopes = list(epitopes_list["Epitope"].unique())
    CITE_DF = importCITE()
    CITE_DF = CITE_DF[epitopes + ["CellType2"]]
    CITE_DF = CITE_DF.rename(columns={"CellType2": "Cell Type"})
    sample_DF = sample_receptor_abundances(
        CITE_DF=CITE_DF,
        numCells=CITE_DF.shape[0],
        targCellType="Treg",
        offTargCellTypes=["Treg"],
    )
    sample_DF = sample_DF.drop(columns="Cell Type")
    samples = sample_DF.to_numpy()
    rec_mean = samples.mean()
    rec_std = samples.std()
    affs = rng.uniform(7, 9, num_receptors)

    dose = 0.1
    recCounts = np.abs(rng.normal(rec_mean, rec_std, (num_cells, num_receptors)))
    valencies = np.array([[1, 1, 1]])
    monomerAffs = restructure_affs(affs)
    Kx_star = 2.24e-12

    R_bound = cyt_binding_model(
        dose=dose, recCounts=recCounts, valencies=valencies, monomerAffs=monomerAffs, Kx_star=Kx_star
    )

    assert R_bound.shape == recCounts.shape


def test_invalid_model_function_inputs():
    rng = np.random.default_rng(1)

    # Test invalid inputs for restructuringAffs
    with pytest.raises(AssertionError):
        restructure_affs(np.array([[8.0, 8.0], [8.0, 8.0]]))  # 2D receptor affinities

    with pytest.raises(AssertionError):
        restructure_affs(np.array([]))  # empty array

    # Assign default values for cytBindingModel and minOffTargSelec
    dose = 0.1
    recCounts1D = rng.uniform(size=3)  # for testing one cell, three receptors
    recCounts2D = rng.uniform(
        size=(100, 3)
    )  # for testing multiple cells, three receptors
    valencies = np.array([[1, 1, 1]])
    monomerAffs = np.array([8.0, 8.0, 8.0])
    modelAffs = restructure_affs(monomerAffs)

    # Test invalid inputs for cytBindingModel
    with pytest.raises(AssertionError):
        cyt_binding_model(
            dose=dose,
            recCounts=rng.uniform(size=(100, 3, 3)),
            valencies=valencies,
            monomerAffs=modelAffs,
            Kx_star=2.24e-12,
        )  # 3D receptor counts

    with pytest.raises(AssertionError):
        cyt_binding_model(
            dose=dose,
            recCounts=recCounts2D,
            valencies=np.array([[1, 1, 1, 1]]),
            monomerAffs=modelAffs,
            Kx_star=2.24e-12,
        )  # wrong number of valencies

    with pytest.raises(AssertionError):
        cyt_binding_model(
            dose=dose,
            recCounts=recCounts2D,
            valencies=valencies,
            monomerAffs=restructure_affs(np.array([8.0, 8.0, 8.0, 8.0])),
            Kx_star=2.24e-12,
        )  # wrong number of complexes

    with pytest.raises(AssertionError):
        cyt_binding_model(
            dose=dose,
            recCounts=recCounts1D,
            valencies=np.array([[1, 1]]),
            monomerAffs=restructure_affs(np.array([8.0, 8.0])),
            Kx_star=2.24e-12,
        )  # 1D mismatched number of receptors

    with pytest.raises(AssertionError):
        cyt_binding_model(
            dose=dose,
            recCounts=recCounts2D,
            valencies=np.array([[1, 1]]),
            monomerAffs=restructure_affs(np.array([8.0, 8.0])),
            Kx_star=2.24e-12,
        )  # 2D mismatched number of receptors

    # Test invalid inputs for minOffTargSelec
    with pytest.raises(AssertionError):
        params = np.concatenate([monomerAffs, [np.log10(2.24e-12)]])
        min_off_targ_selec(
            params=params,
            targRecs=recCounts2D,
            offTargRecs=rng.uniform(size=(100, 4)),
            dose=dose,
            valencies=valencies,
        )  # mismatched number of receptors

    # Test invalid inputs for optimizeSelectivityAffs
    with pytest.raises(AssertionError):
        optimize_affs(
            targRecs=np.array([]),
            offTargRecs=recCounts2D,
            dose=dose,
            valencies=valencies,
        )  # empty target receptors

    # Assign default values for sampleReceptorAbundances
    df = pd.DataFrame(
        {
            "Cell Type": ["Treg"] * 100,
            "CD122": rng.uniform(size=100),
            "CD25": rng.uniform(size=100),
        }
    )
    numCells = 50

    # Test invalid inputs for sampleReceptorAbundances
    with pytest.raises(AssertionError):
        sample_receptor_abundances(
            CITE_DF=df, numCells=200, targCellType="Treg", offTargCellTypes=["Treg"]
        )  # requested sample size greater than available cells

    with pytest.raises(AssertionError):
        sample_receptor_abundances(
            CITE_DF=df.drop(columns="Cell Type"),
            numCells=numCells,
            targCellType="Treg",
            offTargCellTypes=["Treg"],
        )  # lack of cell type column
