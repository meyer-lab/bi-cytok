"""
Functions for calculating binding model selectivity, EMD, and KL divergence for many
receptor combinations and target cell types.
"""

import time

import numpy as np
import pandas as pd

from .distance_metric_funcs import KL_EMD_1D, KL_EMD_2D, KL_EMD_3D
from .imports import sample_receptor_abundances
from .selectivity_funcs import optimize_affs


def _sample_cells(
    rec_abundances: np.ndarray,
    cell_type_labels: np.ndarray,
    targ_cell_type: str,
    sample_size: int = 100,
    balance: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample cells from receptor abundance data.

    Arguments:
        rec_abundances: receptor abundances (rows) across receptors (columns)
        cell_type_labels: cell type label for each cell
        targ_cell_type: target cell type to distinguish from off-target population
        sample_size: total number of cells to sample, split between target and
            off-target populations
        balance: whether to balance number of cells in target and off-target
            populations
    
    Outputs:
        sampled_rec_abundances: subset of rec_abundances after sampling
        sampled_cell_type_labels: subset of cell_type_labels corresponding to
            sampled cells
    """

    sampled_abun_DF = sample_receptor_abundances(
        pd.DataFrame(
            np.hstack((rec_abundances, np.asarray(cell_type_labels).reshape(-1, 1))),
            columns=[f"Rec_{j}" for j in range(rec_abundances.shape[1])]
            + ["Cell Type"],
        ),
        numCells=sample_size,
        targCellType=targ_cell_type,
        balance=balance,
    )
    sampled_cell_type_labels = sampled_abun_DF["Cell Type"].to_numpy(dtype=str)
    sampled_rec_abundances = sampled_abun_DF.drop(columns=["Cell Type"]).to_numpy(
        dtype=float
    )

    return sampled_rec_abundances, sampled_cell_type_labels


def scan_KL_EMD(
    rec_abundances: np.ndarray,
    cell_type_labels: np.ndarray,
    targ_cell_types: list[str],
    dim: int,
    sample_size: int = 100,
    filter_by_target_expr: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate KL divergence and EMD for all receptor combinations across target cell
    types. For each target cell type, samples cells and calculates distribution
    difference metrics between target and off-target populations for all combinations
    of receptors up to the specified dimensionality.

    Arguments:
        rec_abundances: receptor abundances (rows) across receptors (columns)
        cell_type_labels: cell type label for each cell
        targ_cell_types: list of target cell types to evaluate
        dim: dimensionality of receptor combinations (1, 2, or 3)
        sample_size: target cell count for subsampling
        filter_by_target_expr: if True, restrict scan to receptors with higher mean
            expression in target cells than off-target cells. Filtering is applied
            per cell type after sampling, so valid receptors may differ across cell
            types.
    
    Outputs:
        KL_div_vals_scan: KL divergence values for all receptor combinations and cell types
        EMD_vals_scan: EMD values for all receptor combinations and cell types
    """

    assert rec_abundances.shape[0] == len(cell_type_labels)
    assert all([cell_type in cell_type_labels for cell_type in targ_cell_types])
    assert dim in [1, 2, 3]
    assert rec_abundances.shape[1] >= dim

    # Define array shapes based on analysis dimensionality. Arrays contain one
    #   n-dimensional receptor x receptor hypercube per cell type.
    n_receptors = rec_abundances.shape[1]
    output_shape = (n_receptors,) * dim + (len(targ_cell_types),)
    EMD_vals_scan = np.full(output_shape, np.nan)
    KL_div_vals_scan = np.full(output_shape, np.nan)

    for i, cell_type in enumerate(targ_cell_types):
        time_start = time.time()

        # Resample for each target cell type because number of cells per type varies
        sampled_rec_abundances, sampled_cell_type_labels = _sample_cells(
            rec_abundances,
            cell_type_labels,
            targ_cell_type=cell_type,
            sample_size=sample_size,
            balance=True,
        )

        targ_mask = sampled_cell_type_labels == cell_type
        off_targ_mask = ~targ_mask

        # Filters out receptors with higher mean expression in off-target cells.
        # Off-target populations with higher mean expression than target populations
        #    yield high EMD and KL div., but are poor selectivity targets.
        if filter_by_target_expr:
            mean_targ = sampled_rec_abundances[targ_mask, :].mean(axis=0)
            mean_off_targ = sampled_rec_abundances[off_targ_mask, :].mean(axis=0)
            valid_indices = np.where(mean_targ > mean_off_targ)[0]
            filtered_abundances = sampled_rec_abundances[:, valid_indices]
            print(
                f"  Filtered to {len(valid_indices)} / {n_receptors} receptors with "
                f"higher target expression for {cell_type}."
            )
        else:
            valid_indices = np.arange(n_receptors)
            filtered_abundances = sampled_rec_abundances

        if dim == 1:
            kl, emd = KL_EMD_1D(filtered_abundances, targ_mask, off_targ_mask)
            KL_div_vals_scan[valid_indices, i] = kl
            EMD_vals_scan[valid_indices, i] = emd
        elif dim == 2:
            kl, emd = KL_EMD_2D(filtered_abundances, targ_mask, off_targ_mask)
            for li, gi in enumerate(valid_indices):
                KL_div_vals_scan[gi, valid_indices, i] = kl[li, :]
                EMD_vals_scan[gi, valid_indices, i] = emd[li, :]
        else:
            kl, emd = KL_EMD_3D(filtered_abundances, targ_mask, off_targ_mask)
            for li, gi in enumerate(valid_indices):
                for lj, gj in enumerate(valid_indices):
                    KL_div_vals_scan[gi, gj, valid_indices, i] = kl[li, lj, :]
                    EMD_vals_scan[gi, gj, valid_indices, i] = emd[li, lj, :]

        print(
            f"Completed KL/EMD scan for {cell_type} in {time.time() - time_start:.2f} seconds."
        )

    return KL_div_vals_scan, EMD_vals_scan


def scan_selectivity(
    rec_abundances: np.ndarray,
    cell_type_labels: np.ndarray,
    targ_cell_types: list[str],
    dim: int,
    dose: float,
    valencies: np.ndarray,
    sample_size: int = 100,
    signal_col: int = 0,
    init_method: np.ndarray | str | int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimize binding selectivity for all receptor combinations across target cell
    types. For each target cell type, samples cells and optimizes monomer affinities
    and Kx_star to maximize selectivity binding in target populations for
    all combinations of receptors up to the specified dimensionality.

    Arguments:
        rec_abundances: receptor abundances (rows) across receptors (columns)
        cell_type_labels: cell type label for each cell
        targ_cell_types: list of target cell types to evaluate
        dim: dimensionality of receptor combinations (1 or 2)
        dose: ligand concentration/dose in molar units
        valencies: array of valencies for each distinct ligand in the ligand complex.
            Assumes symmetric target receptor valencies for dim > 1.
        sample_size: number of cells to sample per cell type
        signal_col: column index of designated signal receptor to include in all
            combinations, defaults to first receptor.
        init_method: method for initializing optimization (integer seed for random
            initialization, "search" to initialize with grid search, or array of
            initial affinity and Kx_star values)
    
    Outputs:
        selec_vals_scan: optimized selectivity values for all receptor combinations
        opt_affs_scan: optimized monomer affinities in log10(M) for all combinations
        opt_Kx_star_scan: optimized Kx_star values for all combinations
    """

    assert rec_abundances.shape[0] == len(cell_type_labels)
    assert all([cell_type in cell_type_labels for cell_type in targ_cell_types])
    assert dim in [1, 2]
    assert rec_abundances.shape[1] >= dim
    assert len(valencies[0]) == dim + 1
    assert 0 <= signal_col < rec_abundances.shape[1]

    # Define array shapes for each output based on analysis dimensionality. Arrays
    #   contain one n-dimensional receptor x receptor hypercube per cell type. Affinity
    #   array contains an additional dimension, one affinity per evaluated receptor.
    n_receptors = rec_abundances.shape[1]
    output_shape = (n_receptors,) * dim + (len(targ_cell_types),)
    selec_vals_scan = np.full(output_shape, np.nan)
    opt_Kx_star_scan = np.full(output_shape, np.nan)
    affs_output_shape = (n_receptors,) * dim + (len(targ_cell_types), dim + 1,)
    opt_affs_scan = np.full(affs_output_shape, np.nan)

    for i, cell_type in enumerate(targ_cell_types):
        time_start = time.time()

        sampled_rec_abundances, sampled_cell_type_labels = _sample_cells(
            rec_abundances,
            cell_type_labels,
            targ_cell_type=cell_type,
            sample_size=sample_size,
            balance=False, # Binding model is not biased by cell type proportions
        )

        targ_mask = sampled_cell_type_labels == cell_type
        off_targ_mask = ~targ_mask

        # Signal receptor is the same regardless of dimensionality
        signal_rec_abun = np.reshape(sampled_rec_abundances[:, signal_col], (-1, 1))

        if dim == 1:
            for j in range(sampled_rec_abundances.shape[1]):
                rec_abun_pruned = np.reshape(sampled_rec_abundances[:, j], (-1, 1))
                rec_abun_pruned = np.hstack((signal_rec_abun, rec_abun_pruned))
                targ_recs = rec_abun_pruned[targ_mask, :]
                off_targ_recs = rec_abun_pruned[off_targ_mask, :]

                opt_selec, opt_aff_vals, opt_Kx_star = optimize_affs(
                    targ_recs, off_targ_recs, dose, valencies, init_vals=init_method
                )

                selec_vals_scan[j, i] = 1 / opt_selec
                opt_affs_scan[j, i, :] = opt_aff_vals
                opt_Kx_star_scan[j, i] = opt_Kx_star

        if dim == 2:
            time_init = time.time()
            intervals = []

            # Triangular indices assume symmetry across the diagonal which is not
            #   true if valencies are asymmetric
            row, col = np.tril_indices(sampled_rec_abundances.shape[1], k=0)
            for count, (rec1_ind, rec2_ind) in enumerate(zip(row, col, strict=False)):
                rec_abun_pruned = sampled_rec_abundances[:, [rec1_ind, rec2_ind]]

                # When target recptors are the same, they should be modeled as a single
                #   entity with shared valency.
                model_valencies = valencies.copy()
                if rec1_ind == rec2_ind:
                    total_target_valency = valencies[:, 1:3].sum()
                    model_valencies[:, 1] = total_target_valency
                    model_valencies[:, 2] = 0

                rec_abun_pruned = np.hstack((signal_rec_abun, rec_abun_pruned))

                targ_recs = rec_abun_pruned[targ_mask, :]
                off_targ_recs = rec_abun_pruned[off_targ_mask, :]

                try:
                    opt_selec, opt_aff_vals, opt_Kx_star = optimize_affs(
                        targ_recs,
                        off_targ_recs,
                        dose,
                        model_valencies,
                        init_vals=init_method,
                    )
                    selec_vals_scan[rec1_ind, rec2_ind, i] = 1 / opt_selec
                # Optimization occassionally fails for certain ill-conditioned
                #   receptor combinations.
                except Exception as e:
                    print(
                        f"Optimization failed for {cell_type} with receptors {rec1_ind} and {rec2_ind}: {e}"
                    )
                    opt_selec, opt_aff_vals, opt_Kx_star = (
                        np.nan,
                        np.full(rec_abun_pruned.shape[1], np.nan),
                        np.nan,
                    )
                    selec_vals_scan[rec1_ind, rec2_ind, i] = opt_selec

                opt_affs_scan[rec1_ind, rec2_ind, i, :] = opt_aff_vals
                opt_Kx_star_scan[rec1_ind, rec2_ind, i] = opt_Kx_star
                
                # Progress logging:
                if count % 500 == 0:
                    if count == 0:
                        print(
                            f"Compilation time for {cell_type}: {time.time() - time_start:.2f} seconds."
                        )
                    else:
                        intervals.append(time.time() - time_init)
                        print(
                            f"Completed last 500 of {count} out of {len(row)} combinations in {intervals[-1]:.2f} s."
                        )
                        average_interval_per_combo = (
                            sum(intervals) / len(intervals) / 500
                        )
                        estimated_time_remaining = average_interval_per_combo * (
                            len(row) - count
                        )
                        print(
                            f"Estimated time remaining for {cell_type}: {estimated_time_remaining:.2f} seconds."
                        )
                    time_init = time.time()

        print(
            f"Completed selectivity scan for {cell_type} in {time.time() - time_start:.2f} seconds."
        )

    return selec_vals_scan, opt_affs_scan, opt_Kx_star_scan