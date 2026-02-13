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


def _setup_starting_point_test_data(
    sample_size, cell_type, ill_conditioned_recs, dose, valencies
):
    """Helper function to set up common test data for starting point methods."""
    CITE_DF = importCITE()
    CITE_DF = CITE_DF.rename(columns={"CellType2": "Cell Type"})
    CITE_DF = CITE_DF.drop(columns="CellType1")
    CITE_DF = CITE_DF.drop(columns="CellType3")
    sample_DF = sample_receptor_abundances(
        CITE_DF=CITE_DF,
        numCells=sample_size,
        targCellType=cell_type,
    )
    targ_mask = (sample_DF["Cell Type"] == cell_type).to_numpy()

    ill_posed_params = [12, 12, 12, -15]
    low_selec = (
        1
        / optimize_affs(
            targRecs=sample_DF[ill_conditioned_recs].to_numpy()[targ_mask],
            offTargRecs=sample_DF[ill_conditioned_recs].to_numpy()[~targ_mask],
            dose=dose,
            valencies=valencies,
            init_vals=ill_posed_params,
        )[0]
    )
    print(f"Ill-conditioned optimization yields selectivity: {low_selec}")

    return sample_DF, targ_mask, low_selec


def test_rand_init():
    """
    Test that random initialization starting point method does not fail for too many receptor combinations.

    Frequency threshold is representative of the current state of the method
    and may need to be adjusted as the method improves/worsens.
    """
    SAMPLE_SIZE = 1000
    DOSE = 1e-10
    VALENCIES = np.array([[2, 1, 1]])
    CELL_TYPE = "Treg"
    TARG_REC = "CD122"
    ILL_CONDITIONED_RECS = ["CD122", "CD338", "CD45RA"]
    TEST_RECS = ["CD338", "CD45RA", "CD25", "CD4-1", "CD28", "CD278"]
    N_RANDS = 5
    FAILED_SELEC_CUTOFF = 0.005
    RANDOM_INIT_FAIL_THRESHOLD = 0.6

    sample_DF, targ_mask, low_selec = _setup_starting_point_test_data(
        SAMPLE_SIZE, CELL_TYPE, ILL_CONDITIONED_RECS, DOSE, VALENCIES
    )

    random_selec_list = []
    row, col = np.tril_indices(len(TEST_RECS), k=0)
    for rand_state in range(N_RANDS):
        for i, j in zip(row, col, strict=False):
            rec1 = TEST_RECS[i]
            rec2 = TEST_RECS[j]
            model_recs = [TARG_REC, rec1, rec2]
            random_selec = (
                1
                / optimize_affs(
                    targRecs=sample_DF[model_recs].to_numpy()[targ_mask],
                    offTargRecs=sample_DF[model_recs].to_numpy()[~targ_mask],
                    dose=DOSE,
                    valencies=VALENCIES,
                    init_vals=rand_state,
                )[0]
            )
            random_selec_list.append(random_selec)
    rand_fail_freq = sum(
        selec <= low_selec + FAILED_SELEC_CUTOFF for selec in random_selec_list
    ) / len(random_selec_list)
    print(f"Random initialization failure frequency: {rand_fail_freq}")
    assert rand_fail_freq < RANDOM_INIT_FAIL_THRESHOLD


def test_search_init():
    """
    Test that initialization search starting point method does not fail for too many receptor combinations.
    """
    SAMPLE_SIZE = 1000
    DOSE = 1e-10
    VALENCIES = np.array([[2, 1, 1]])
    CELL_TYPE = "Treg"
    TARG_REC = "CD122"
    ILL_CONDITIONED_RECS = ["CD122", "CD338", "CD45RA"]
    TEST_RECS = ["CD338", "CD45RA", "CD25", "CD4-1", "CD28", "CD278"]
    FAILED_SELEC_CUTOFF = 0.005
    INIT_SEARCH_FAIL_THRESHOLD = 0.1

    sample_DF, targ_mask, low_selec = _setup_starting_point_test_data(
        SAMPLE_SIZE, CELL_TYPE, ILL_CONDITIONED_RECS, DOSE, VALENCIES
    )

    init_search_selec_list = []
    row, col = np.tril_indices(len(TEST_RECS), k=0)
    for i, j in zip(row, col, strict=False):
        rec1 = TEST_RECS[i]
        rec2 = TEST_RECS[j]
        model_recs = [TARG_REC, rec1, rec2]
        init_search_selec = (
            1
            / optimize_affs(
                targRecs=sample_DF[model_recs].to_numpy()[targ_mask],
                offTargRecs=sample_DF[model_recs].to_numpy()[~targ_mask],
                dose=DOSE,
                valencies=VALENCIES,
                init_vals="search",
            )[0]
        )
        init_search_selec_list.append(init_search_selec)
    search_fail_freq = sum(
        selec <= low_selec + FAILED_SELEC_CUTOFF for selec in init_search_selec_list
    ) / len(init_search_selec_list)
    print(f"Initialization search failure frequency: {search_fail_freq}")
    assert search_fail_freq < INIT_SEARCH_FAIL_THRESHOLD


def test_fixed_init():
    """
    Test that fixed starting point method does not fail for too many receptor combinations.
    """
    SAMPLE_SIZE = 1000
    DOSE = 1e-10
    VALENCIES = np.array([[2, 1, 1]])
    CELL_TYPE = "Treg"
    TARG_REC = "CD122"
    ILL_CONDITIONED_RECS = ["CD122", "CD338", "CD45RA"]
    TEST_RECS = ["CD338", "CD45RA", "CD25", "CD4-1", "CD28", "CD278"]
    FIXED_STARTING_POINT = [6.0, 7.0, 7.0, -9.0]
    FAILED_SELEC_CUTOFF = 0.005
    FIXED_START_FAIL_THRESHOLD = 0.05

    sample_DF, targ_mask, low_selec = _setup_starting_point_test_data(
        SAMPLE_SIZE, CELL_TYPE, ILL_CONDITIONED_RECS, DOSE, VALENCIES
    )

    fixed_selec_list = []
    row, col = np.tril_indices(len(TEST_RECS), k=0)
    for i, j in zip(row, col, strict=False):
        rec1 = TEST_RECS[i]
        rec2 = TEST_RECS[j]
        model_recs = [TARG_REC, rec1, rec2]
        fixed_selec = (
            1
            / optimize_affs(
                targRecs=sample_DF[model_recs].to_numpy()[targ_mask],
                offTargRecs=sample_DF[model_recs].to_numpy()[~targ_mask],
                dose=DOSE,
                valencies=VALENCIES,
                init_vals=FIXED_STARTING_POINT,
            )[0]
        )
        fixed_selec_list.append(fixed_selec)
    fixed_fail_freq = sum(
        selec <= low_selec + FAILED_SELEC_CUTOFF for selec in fixed_selec_list
    ) / len(fixed_selec_list)
    print(f"Fixed starting point failure frequency: {fixed_fail_freq}")
    assert fixed_fail_freq < FIXED_START_FAIL_THRESHOLD


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
