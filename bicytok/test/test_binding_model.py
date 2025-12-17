"""
Unit test file for binding model functions.
"""

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
