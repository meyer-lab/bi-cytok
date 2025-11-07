"""
Functions for calculating binding model selectivity, EMD, and KL divergence for many
receptor combinations and target cell types.
"""
import time

import pandas as pd
import numpy as np

from .distance_metric_funcs import KL_EMD_1D, KL_EMD_2D, KL_EMD_3D
from .imports import sample_receptor_abundances
from .selectivity_funcs import optimize_affs


def scan_KL_EMD(
    rec_abundances: np.ndarray,
    cell_type_labels: np.ndarray,
    targ_cell_types: list,
    dim: int,
    sample_size: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """ """

    assert rec_abundances.shape[0] == len(cell_type_labels)
    assert all([cell_type in cell_type_labels for cell_type in targ_cell_types])
    assert dim in [1, 2, 3]
    assert rec_abundances.shape[1] >= dim

    output_shape = (rec_abundances.shape[1],) * dim + (len(targ_cell_types),)
    EMD_vals_scan = np.full(output_shape, np.nan)
    KL_div_vals_scan = np.full(output_shape, np.nan)

    for i, cell_type in enumerate(targ_cell_types):
        time_start = time.time()
        sampled_abun_DF = sample_receptor_abundances(
            pd.DataFrame(np.hstack((rec_abundances, np.array(cell_type_labels)[:, None])), columns=
                            [f"Rec_{j}" for j in range(rec_abundances.shape[1])] + ["Cell Type"]),
            numCells=sample_size,
            targCellType=cell_type,
            balance=True
        )
        sampled_cell_type_labels = sampled_abun_DF["Cell Type"].to_numpy(dtype=str)
        sampled_rec_abundances = sampled_abun_DF.drop(columns=["Cell Type"]).to_numpy(dtype=float)

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
        print(f"Completed KL/EMD scan for {cell_type} in {time.time() - time_start:.2f} seconds.")

    return KL_div_vals_scan, EMD_vals_scan


def scan_selectivity(
    rec_abundances: np.ndarray,
    cell_type_labels: np.ndarray,
    targ_cell_types: list,
    dim: int,
    dose: float,
    valencies: np.ndarray,
    sample_size: int = 100,
    signal_col: int | None = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ """

    assert rec_abundances.shape[0] == len(cell_type_labels)
    assert all([cell_type in cell_type_labels for cell_type in targ_cell_types])
    assert dim in [1, 2, 3]
    assert rec_abundances.shape[1] >= dim
    if signal_col is not None:
        assert len(valencies[0]) == dim + 1
    else:
        assert len(valencies[0]) == dim
    assert signal_col is None or (0 <= signal_col < rec_abundances.shape[1])

    output_shape = (rec_abundances.shape[1],) * dim + (len(targ_cell_types),)
    selec_vals_scan = np.full(output_shape, np.nan)
    opt_Kx_star_scan = np.full(output_shape, np.nan)
    affs_output_shape = (rec_abundances.shape[1],) * dim + (len(targ_cell_types), dim + (1 if signal_col is not None else 0))
    opt_affs_scan = np.full(affs_output_shape, np.nan)

    for i, cell_type in enumerate(targ_cell_types):
        sampled_abun_DF = sample_receptor_abundances(
            pd.DataFrame(np.hstack((rec_abundances, np.array(cell_type_labels)[:, None])), columns=
                            [f"Rec_{j}" for j in range(rec_abundances.shape[1])] + ["Cell Type"]),
            numCells=sample_size,
            targCellType=cell_type,
            balance=False # Balanced distributions not necessary for binding model (I think)
        )
        sampled_cell_type_labels = sampled_abun_DF["Cell Type"].to_numpy(dtype=str)
        sampled_rec_abundances = sampled_abun_DF.drop(columns=["Cell Type"]).to_numpy(dtype=float)

        targ_mask = sampled_cell_type_labels == cell_type
        off_targ_mask = ~targ_mask

        if signal_col is not None:
            signal_rec_abun = np.reshape(sampled_rec_abundances[:, signal_col], (-1, 1))

        opt_selecs = np.full((sampled_rec_abundances.shape[1],) * dim, np.nan)
        if signal_col is not None:
            opt_affs = np.full((sampled_rec_abundances.shape[1],) * dim + (dim + 1,), np.nan)
        else:
            opt_affs = np.full((sampled_rec_abundances.shape[1],) * dim + (dim,), np.nan)
        opt_Kx_stars = np.full((sampled_rec_abundances.shape[1],) * dim, np.nan)
        if dim == 1:
            # opt_selecs = np.full((sampled_rec_abundances.shape[1]), np.nan)
            # if signal_col is not None:
            #     opt_affs = np.full((sampled_rec_abundances.shape[1], dim + 1), np.nan)
            # else:
            #     opt_affs = np.full((sampled_rec_abundances.shape[1], dim), np.nan)
            # opt_Kx_stars = np.full((sampled_rec_abundances.shape[1]), np.nan)

            for j in range(sampled_rec_abundances.shape[1]):
                rec_abun_pruned = np.reshape(sampled_rec_abundances[:, j], (-1, 1))

                if signal_col is not None:
                    rec_abun_pruned = np.hstack((signal_rec_abun, rec_abun_pruned))

                targ_recs = rec_abun_pruned[targ_mask, :]
                off_targ_recs = rec_abun_pruned[off_targ_mask, :]

                opt_selecs[j], opt_affs[j, :], opt_Kx_stars[j] = optimize_affs(
                    targ_recs, off_targ_recs, dose, valencies
                )
                opt_selecs[j] = 1 / opt_selecs[j]

            selec_vals_scan[:, i] = opt_selecs
            opt_affs_scan[:, i, :] = opt_affs
            opt_Kx_star_scan[:, i] = opt_Kx_stars

        if dim == 2:
            # opt_selecs = np.full((sampled_rec_abundances.shape[0],) * dim, np.nan)
            # opt_affs = np.full((sampled_rec_abundances.shape[0],) * dim, np.nan)
            # opt_Kx_stars = np.full((sampled_rec_abundances.shape[0],) * dim, np.nan)

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

                (
                    opt_selecs[rec1_ind, rec2_ind],
                    opt_affs[rec1_ind, rec2_ind, :],
                    opt_Kx_stars[rec1_ind, rec2_ind],
                ) = optimize_affs(targ_recs, off_targ_recs, dose, valencies)
                opt_selecs[rec1_ind, rec2_ind] = 1 / opt_selecs[rec1_ind, rec2_ind]

            selec_vals_scan[:, :, i] = opt_selecs
            opt_affs_scan[:, :, i, :] = opt_affs
            opt_Kx_star_scan[:, :, i] = opt_Kx_stars

        if dim == 3:
            pass

    return selec_vals_scan, opt_affs_scan, opt_Kx_star_scan
