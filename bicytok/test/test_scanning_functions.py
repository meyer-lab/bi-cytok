"""
Unit test file for receptor scanning functions.
"""

import random
from unittest.mock import patch

import numpy as np
import pytest

from figures.scanning_analyses.scan_runners import (
    load_KL_EMD_scan_results,
    load_selec_scan_results,
    run_KL_EMD_scan,
    run_selectivity_scan,
)

from ..imports import sample_test_data as sample_data
from ..scanning_funcs import scan_KL_EMD, scan_selectivity


def test_selectivity_scan():
    """
    Test for reasonable output shapes from binding model selectivity scanning function.
    """

    n_cells = 250
    n_receptors = 3
    recAbundances, _, _ = sample_data(n_obs=n_cells, n_var=n_receptors)
    cell_type_labels = [
        random.choice(["A", "B", "C", "D"]) for _ in range(recAbundances.shape[0])
    ]
    targ_cells = ["A", "B", "C"]
    dose = 0.1
    sample_size = n_cells

    # Test single-receptor targeting
    dim = 1
    valencies = np.array([[1, 1]])
    opt_selec, opt_affs, opt_kx_star = scan_selectivity(
        rec_abundances=recAbundances,
        cell_type_labels=cell_type_labels,
        targ_cell_types=targ_cells,
        dim=dim,
        dose=dose,
        valencies=valencies,
        sample_size=sample_size,
    )

    assert opt_selec.shape == (n_receptors, len(targ_cells))
    assert opt_affs.shape == (n_receptors, len(targ_cells), len(valencies[0]))
    assert opt_kx_star.shape == (n_receptors, len(targ_cells))

    # Test pair-wise targeting
    dim = 2
    valencies = np.array([[1, 1, 1]])
    opt_selec, opt_affs, opt_kx_star = scan_selectivity(
        rec_abundances=recAbundances,
        cell_type_labels=cell_type_labels,
        targ_cell_types=targ_cells,
        dim=dim,
        dose=dose,
        valencies=valencies,
        sample_size=sample_size,
    )

    assert opt_selec.shape == (n_receptors, n_receptors, len(targ_cells))
    assert opt_affs.shape == (
        n_receptors,
        n_receptors,
        len(targ_cells),
        len(valencies[0]),
    )
    assert opt_kx_star.shape == (n_receptors, n_receptors, len(targ_cells))


def test_KL_EMD_scan():
    """
    Test for reasonable output types and shapes from KL div./EMD scanning functions.
    """

    n_cells = 250
    n_receptors = 3
    recAbundances, _, _ = sample_data(n_obs=n_cells, n_var=n_receptors)
    cell_type_labels = [
        random.choice(["A", "B", "C", "D"]) for _ in range(recAbundances.shape[0])
    ]
    targ_cells = ["A", "B", "C"]
    sample_size = n_cells

    # Test single-receptor targeting
    dim = 1
    KL_res, EMD_res = scan_KL_EMD(
        rec_abundances=recAbundances,
        cell_type_labels=cell_type_labels,
        targ_cell_types=targ_cells,
        dim=dim,
        sample_size=sample_size,
    )

    assert KL_res.shape == (n_receptors, len(targ_cells))
    assert EMD_res.shape == (n_receptors, len(targ_cells))

    # Test single-receptor targeting
    dim = 2
    KL_res, EMD_res = scan_KL_EMD(
        rec_abundances=recAbundances,
        cell_type_labels=cell_type_labels,
        targ_cell_types=targ_cells,
        dim=dim,
        sample_size=sample_size,
    )

    assert KL_res.shape == (n_receptors, n_receptors, len(targ_cells))
    assert EMD_res.shape == (n_receptors, n_receptors, len(targ_cells))


def test_invalid_inputs():
    """Test for appropriate error handling of invalid inputs in scanning functions."""

    n_cells = 250
    n_receptors = 3
    recAbundances, _, _ = sample_data(n_obs=n_cells, n_var=n_receptors)
    cell_type_labels = [
        random.choice(["A", "B", "C", "D"]) for _ in range(recAbundances.shape[0])
    ]
    targ_cells = ["A", "B", "C"]
    dose = 0.1
    sample_size = n_cells
    dim = 1
    valencies = np.array([[1, 1]])

    ## Test invalid inputs for selectivity scan
    with pytest.raises(AssertionError):
        scan_selectivity(
            rec_abundances=recAbundances,
            cell_type_labels=cell_type_labels[0:100],
            targ_cell_types=targ_cells,
            dim=dim,
            dose=dose,
            valencies=valencies,
            sample_size=sample_size,
        )  # Mismatched cell type label and receptor abundance lengths

    with pytest.raises(AssertionError):
        scan_selectivity(
            rec_abundances=recAbundances,
            cell_type_labels=cell_type_labels,
            targ_cell_types=["A", "B", "C", "E"],
            dim=dim,
            dose=dose,
            valencies=valencies,
            sample_size=sample_size,
        )  # Target cell type not in cell type labels

    with pytest.raises(AssertionError):
        scan_selectivity(
            rec_abundances=recAbundances,
            cell_type_labels=cell_type_labels,
            targ_cell_types=targ_cells,
            dim=dim,
            dose=dose,
            valencies=np.array([[1, 1, 1]]),
            sample_size=sample_size,
        )  # Valencies not compatible with dimensionality

    dim = 2
    valencies = np.array([[1, 1, 2]])
    with pytest.warns(UserWarning):
        scan_selectivity(
            rec_abundances=recAbundances,
            cell_type_labels=cell_type_labels,
            targ_cell_types=targ_cells,
            dim=dim,
            dose=dose,
            valencies=valencies,
            sample_size=sample_size,
            asym_targs=False,
        )  # dim==2, asym_targs==False with asymmetric valencies should trigger warning

    ## Test invalid inputs for KL/EMD scan
    with pytest.raises(AssertionError):
        scan_KL_EMD(
            rec_abundances=recAbundances,
            cell_type_labels=cell_type_labels[0:100],
            targ_cell_types=targ_cells,
            dim=dim,
            sample_size=sample_size,
        )  # Mismatched cell type label and receptor abundance lengths

    with pytest.raises(AssertionError):
        scan_KL_EMD(
            rec_abundances=recAbundances,
            cell_type_labels=cell_type_labels,
            targ_cell_types=["A", "B", "C", "E"],
            dim=dim,
            sample_size=sample_size,
        )  # Target cell type not in cell type labels


def test_runner_loader_roundtrip(tmp_path):
    """
    Test that scan result loader functions reconstruct the same arrays saved by
    runner functions. The underlying scan computations are replaced with random arrays
    of the correct shape so the test runs quickly and without triggering assertions
    inside the scan functions that require all hardcoded cell types to be present in
    the data.
    """

    rng = np.random.default_rng(0)

    def mock_scan_selectivity(
        rec_abundances,  # noqa: ARG001
        cell_type_labels,  # noqa: ARG001
        targ_cell_types,  # noqa: ARG001
        dim,  # noqa: ARG001
        valencies,  # noqa: ARG001
        **kwargs,  # noqa: ARG001
    ):
        n = rec_abundances.shape[1]
        k = len(targ_cell_types)
        return (
            rng.random((n, n, k)),
            rng.random((n, n, k, len(valencies[0]))),
            rng.random((n, n, k)),
        )

    def mock_scan_KL_EMD(
        rec_abundances,  # noqa: ARG001
        cell_type_labels,  # noqa: ARG001
        targ_cell_types,  # noqa: ARG001
        dim,  # noqa: ARG001
        **kwargs,  # noqa: ARG001
    ):
        n = rec_abundances.shape[1]
        k = len(targ_cell_types)
        return rng.random((n, n, k)), rng.random((n, n, k))

    ## Testing selectivity scan runner and loader
    selec_path = str(tmp_path / "selec_scan.csv")  # Save to/load from pytest tmp dir

    # Within the runner, replace scanning function with mock scanning function that
    #   generates random arrays of the same shape as the real scanning function
    #   outputs. This bypasses the expensive scanning optimizations.
    with (
        patch(
            "figures.scanning_analyses.scan_runners.scan_selectivity",
            side_effect=mock_scan_selectivity,
        ),
        patch("sys.argv", ["selectivity_scan", "--output-path", selec_path]),
    ):
        orig_selec, orig_affs, orig_kx, orig_recs, orig_cts = run_selectivity_scan()

    loaded_selec, loaded_affs, loaded_kx, loaded_recs, loaded_cts = (
        load_selec_scan_results(selec_path)
    )

    # Loader sorts receptors and cell types alphabetically; map to runner's ordering
    rec_map = [loaded_recs.index(r) for r in orig_recs]
    ct_map = [loaded_cts.index(ct) for ct in orig_cts]
    n_affs = orig_affs.shape[-1]

    np.testing.assert_allclose(
        orig_selec, loaded_selec[np.ix_(rec_map, rec_map, ct_map)], equal_nan=True
    )
    np.testing.assert_allclose(
        orig_kx, loaded_kx[np.ix_(rec_map, rec_map, ct_map)], equal_nan=True
    )
    np.testing.assert_allclose(
        orig_affs,
        loaded_affs[np.ix_(rec_map, rec_map, ct_map, range(n_affs))],
        equal_nan=True,
    )

    ## Testing KL/EMD scan runner and loader
    kl_emd_path = str(tmp_path / "kl_emd_scan.csv")
    with (
        patch(
            "figures.scanning_analyses.scan_runners.scan_KL_EMD",
            side_effect=mock_scan_KL_EMD,
        ),
        patch("sys.argv", ["KL_EMD_scan", "--output-path", kl_emd_path]),
    ):
        orig_kl, orig_emd, orig_recs_kl, orig_cts_kl = run_KL_EMD_scan()
    loaded_kl, loaded_emd, loaded_recs_kl, loaded_cts_kl = load_KL_EMD_scan_results(
        kl_emd_path
    )
    rec_map_kl = [loaded_recs_kl.index(r) for r in orig_recs_kl]
    ct_map_kl = [loaded_cts_kl.index(ct) for ct in orig_cts_kl]
    np.testing.assert_allclose(
        orig_kl, loaded_kl[np.ix_(rec_map_kl, rec_map_kl, ct_map_kl)], equal_nan=True
    )
    np.testing.assert_allclose(
        orig_emd, loaded_emd[np.ix_(rec_map_kl, rec_map_kl, ct_map_kl)], equal_nan=True
    )
