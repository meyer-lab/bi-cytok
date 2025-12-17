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
    Sample cells from receptor abundance data based on target and off-target
    populations. Attempts to balance the number of target and off-target cells, ensures
    balance if specified.

    Arguments:
        rec_abundances: receptor abundances (rows) across receptors (columns)
        cell_type_labels: cell type label for each cell
        targ_cell_type: target cell type to sample
        sample_size: number of cells to sample
        balance: whether to balance target and off-target population proportions

    Outputs:
        sampled_rec_abundances: sampled receptor abundances
        sampled_cell_type_labels: cell type labels for sampled cells
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
        sample_size: number of cells to sample per cell type

    Outputs:
        KL_div_vals_scan: KL divergence values for all receptor combinations and cell types
        EMD_vals_scan: EMD values for all receptor combinations and cell types
    """

    assert rec_abundances.shape[0] == len(cell_type_labels)
    assert all([cell_type in cell_type_labels for cell_type in targ_cell_types])
    assert dim in [1, 2, 3]
    assert rec_abundances.shape[1] >= dim

    n_receptors = rec_abundances.shape[1]

    output_shape = (n_receptors,) * dim + (len(targ_cell_types),)
    EMD_vals_scan = np.full(output_shape, np.nan)
    KL_div_vals_scan = np.full(output_shape, np.nan)

    for i, cell_type in enumerate(targ_cell_types):
        time_start = time.time()
        sampled_rec_abundances, sampled_cell_type_labels = _sample_cells(
            rec_abundances,
            cell_type_labels,
            targ_cell_type=cell_type,
            sample_size=sample_size,
            balance=True,
        )

        targ_mask = sampled_cell_type_labels == cell_type
        off_targ_mask = ~targ_mask

        if dim == 1:
            KL_div_vals_scan[:, i], EMD_vals_scan[:, i] = KL_EMD_1D(
                sampled_rec_abundances, targ_mask, off_targ_mask
            )
        elif dim == 2:
            KL_div_vals_scan[:, :, i], EMD_vals_scan[:, :, i] = KL_EMD_2D(
                sampled_rec_abundances, targ_mask, off_targ_mask
            )
        else:
            KL_div_vals_scan[:, :, :, i], EMD_vals_scan[:, :, :, i] = KL_EMD_3D(
                sampled_rec_abundances, targ_mask, off_targ_mask
            )
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
    signal_col: int | None = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimize binding selectivity for all receptor combinations across target cell
    types. For each target cell type, samples cells and optimizes monomer affinities
    and Kx_star to maximize selectivity between target and off-target populations for
    all combinations of receptors up to the specified dimensionality. A designated
    signal receptor can be included in all combinations, otherwise the first receptor
    in each combination is treated as the signal receptor.

    Arguments:
        rec_abundances: receptor abundances (rows) across receptors (columns)
        cell_type_labels: cell type label for each cell
        targ_cell_types: list of target cell types to evaluate
        dim: dimensionality of receptor combinations (1, 2, or 3)
        dose: ligand concentration/dose in molar units
        valencies: array of valencies for each distinct ligand in the ligand complex
        sample_size: number of cells to sample per cell type
        signal_col: column index of designated signal receptor to include in all
            combinations, or None to exclude signal receptor

    Outputs:
        selec_vals_scan: optimized selectivity values for all receptor combinations
        opt_affs_scan: optimized monomer affinities in log10(M) for all combinations
        opt_Kx_star_scan: optimized Kx_star values for all combinations
    """

    assert rec_abundances.shape[0] == len(cell_type_labels)
    assert all([cell_type in cell_type_labels for cell_type in targ_cell_types])
    assert dim in [1, 2, 3]
    assert rec_abundances.shape[1] >= dim
    assert len(valencies[0]) == dim + (1 if signal_col is not None else 0)
    assert signal_col is None or (0 <= signal_col < rec_abundances.shape[1])

    n_receptors = rec_abundances.shape[1]

    output_shape = (n_receptors,) * dim + (len(targ_cell_types),)
    selec_vals_scan = np.full(output_shape, np.nan)
    opt_Kx_star_scan = np.full(output_shape, np.nan)
    affs_output_shape = (n_receptors,) * dim + (
        len(targ_cell_types),
        dim + (1 if signal_col is not None else 0),
    )
    opt_affs_scan = np.full(affs_output_shape, np.nan)

    for i, cell_type in enumerate(targ_cell_types):
        time_start = time.time()

        sampled_rec_abundances, sampled_cell_type_labels = _sample_cells(
            rec_abundances,
            cell_type_labels,
            targ_cell_type=cell_type,
            sample_size=sample_size,
            balance=False,  # Balanced distributions not necessary for binding model
        )

        targ_mask = sampled_cell_type_labels == cell_type
        off_targ_mask = ~targ_mask

        if signal_col is not None:
            signal_rec_abun = np.reshape(sampled_rec_abundances[:, signal_col], (-1, 1))

        if dim == 1:
            for j in range(sampled_rec_abundances.shape[1]):
                rec_abun_pruned = np.reshape(sampled_rec_abundances[:, j], (-1, 1))

                if signal_col is not None:
                    rec_abun_pruned = np.hstack((signal_rec_abun, rec_abun_pruned))

                targ_recs = rec_abun_pruned[targ_mask, :]
                off_targ_recs = rec_abun_pruned[off_targ_mask, :]

                opt_selec, opt_aff_vals, opt_Kx_star = optimize_affs(
                    targ_recs, off_targ_recs, dose, valencies
                )

                selec_vals_scan[j, i] = 1 / opt_selec
                opt_affs_scan[j, i, :] = opt_aff_vals
                opt_Kx_star_scan[j, i] = opt_Kx_star

        if dim == 2:
            # Triangular indices assume symmetry across the diagonal which is not
            #   true if there is no designated signal receptor or if valencies are
            #   asymmetric
            row, col = np.tril_indices(sampled_rec_abundances.shape[1], k=0)
            for rec1_ind, rec2_ind in zip(row, col, strict=False):
                rec_abun_pruned = sampled_rec_abundances[:, [rec1_ind, rec2_ind]]

                if signal_col is not None:
                    rec_abun_pruned = np.hstack((signal_rec_abun, rec_abun_pruned))

                targ_recs = rec_abun_pruned[targ_mask, :]
                off_targ_recs = rec_abun_pruned[off_targ_mask, :]

                opt_selec, opt_aff_vals, opt_Kx_star = optimize_affs(
                    targ_recs, off_targ_recs, dose, valencies
                )

                selec_vals_scan[rec1_ind, rec2_ind, i] = 1 / opt_selec
                opt_affs_scan[rec1_ind, rec2_ind, i, :] = opt_aff_vals
                opt_Kx_star_scan[rec1_ind, rec2_ind, i] = opt_Kx_star

        if dim == 3:
            pass

        print(
            f"Completed selectivity scan for {cell_type} in {time.time() - time_start:.2f} seconds."
        )

    return selec_vals_scan, opt_affs_scan, opt_Kx_star_scan


# def scan_selectivity_signal(
#     rec_abundances: np.ndarray,
#     cell_type_labels: np.ndarray,
#     targ_cell_type: str,
#     dose: float,
#     valencies: np.ndarray,
#     signal_rec_inds: list[int] | None = None,
#     sample_size: int = 100,
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Optimize binding selectivity for all receptor pairs across all possible signal
#     receptor designations for a single target cell type.
#     """

#     n_receptors = rec_abundances.shape[1]

#     if signal_rec_inds == None:
#         signal_rec_inds = range(n_receptors)

#     output_shape = (n_receptors, n_receptors, len(signal_rec_inds))
#     selec_vals_scan = np.full(output_shape, np.nan)
#     opt_Kx_star_scan = np.full(output_shape, np.nan)
#     affs_output_shape = (n_receptors, n_receptors, len(signal_rec_inds), 3)
#     opt_affs_scan = np.full(affs_output_shape, np.nan)

#     for signal_ind in
