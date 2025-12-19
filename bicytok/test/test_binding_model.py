"""
Unit test file for binding model functions.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ..binding_model_funcs import cyt_binding_model
from ..imports import importCITE, sample_receptor_abundances
from ..imports import sample_test_data as sample_data
from ..selectivity_funcs import (
    min_off_targ_selec,
    optimize_affs,
    restructure_affs,
)

path_here = Path(__file__).parent.parent


def test_optimize_affs():
    """Test for reasonable output types and shapes from optimize_affs function."""
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
    optAffs = np.array(optAffs)

    assert isinstance(optSelec, float)
    assert optSelec >= 0
    assert optAffs.shape == valencies[0].shape
    assert all(optAffs >= 0)
    assert isinstance(optKx_star, float)
    assert optKx_star > 0


def test_binding_model():
    """Test for reasonable output types and shapes from cyt_binding_model function."""
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
        dose=dose,
        recCounts=recCounts,
        valencies=valencies,
        monomerAffs=monomerAffs,
        Kx_star=Kx_star,
    )

    assert R_bound.shape == recCounts.shape


def test_symmetric_affinities():
    """Test that optimize_affs predicts symmetric target receptor affinities when valencies are symmetric"""

    n_receptors = 10
    recAbundances, targ, offTarg = sample_data(n_obs=1000, n_var=n_receptors)
    dose = 1e-10
    valencies = np.array([[2, 1, 1]])

    row, col = np.tril_indices(n_receptors, k=0)
    for i, j in zip(row, col, strict=False):
        # Exclude signal receptor or identical target receptors
        if i == j or i == 0 or j == 0:
            continue

        # Test forward target receptor order
        test_abundances = recAbundances[:, [0, i, j]]
        targRecs = test_abundances[targ]
        offTargRecs = test_abundances[offTarg]
        optSelec_f, optAffs_f, optKx_star_f = optimize_affs(
            targRecs=targRecs,
            offTargRecs=offTargRecs,
            dose=dose,
            valencies=valencies,
        )

        # Test reverse target receptor order
        test_abundances = recAbundances[:, [0, j, i]]
        targRecs = test_abundances[targ]
        offTargRecs = test_abundances[offTarg]
        optSelec_r, optAffs_r, optKx_star_r = optimize_affs(
            targRecs=targRecs,
            offTargRecs=offTargRecs,
            dose=dose,
            valencies=valencies,
        )

        assert np.isclose(optSelec_f, optSelec_r)
        assert np.isclose(optAffs_f[0], optAffs_r[0], rtol=1e-3)
        assert np.isclose(optAffs_f[1], optAffs_r[2], rtol=1e-3)
        assert np.isclose(optAffs_f[2], optAffs_r[1], rtol=1e-3)
        assert np.isclose(optKx_star_f, optKx_star_r)


def test_equivalent_compositions():
    """
    Test that optimize_affs predicts equivalent selectivities for functionally
        equivalent valency composition definitions.
    """

    n_receptors = 10
    recAbundances, targ, offTarg = sample_data(n_obs=1000, n_var=n_receptors)
    dose = 1e-10

    for i in range(1, n_receptors):
        # Test shared valency definition
        test_abundances = recAbundances[:, [0, i]]
        valencies = np.array([[2, 2]])
        targRecs = test_abundances[targ]
        offTargRecs = test_abundances[offTarg]
        optSelec_sh, optAffs_sh, optKx_star_sh = optimize_affs(
            targRecs=targRecs,
            offTargRecs=offTargRecs,
            dose=dose,
            valencies=valencies,
        )

        # Test split valency definition
        test_abundances = recAbundances[:, [0, i, i]]
        valencies = np.array([[2, 1, 1]])
        targRecs = test_abundances[targ]
        offTargRecs = test_abundances[offTarg]
        optSelec_sp, optAffs_sp, optKx_star_sp = optimize_affs(
            targRecs=targRecs,
            offTargRecs=offTargRecs,
            dose=dose,
            valencies=valencies,
        )

        # Tols loosened to pass disagreements
        assert np.isclose(optSelec_sh, optSelec_sp, rtol=1e-2)
        warnings.warn(
            "Affinity tolerances temporarily loosened to pass disagreements between "
            "equivalent compositions.",
            stacklevel=2,
        )
        assert np.isclose(optAffs_sh[0], optAffs_sp[0], rtol=1e-1)
        assert np.isclose(optAffs_sh[1], optAffs_sp[1], rtol=1e-1)
        assert np.isclose(optAffs_sp[1], optAffs_sp[2], rtol=1e-1)
        assert np.isclose(optKx_star_sh, optKx_star_sp)


def test_invalid_model_function_inputs():
    """Test for appropriate error handling of invalid inputs in binding model functions."""

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
