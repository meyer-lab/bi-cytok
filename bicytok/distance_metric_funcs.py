import time
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import ot
import pandas as pd
from scipy import stats
from sklearn.neighbors import KernelDensity

path_here = Path(__file__).parent.parent


KDE_A_TOL = 1e-3  # Tolerance for KDE fitting
KDE_R_TOL = 1e-3  # Tolerance for KDE fitting
BW_METHOD = "scott"  # Bandwidth method for KDE
KDE_GRID_MARGIN = 0.5  # Margin added to min/max for KDE grid
KDE_GRID_SIZE = 10  # Number of points per dimension in KDE grid
ENTROPY_EPS = 1e-6  # Small value to avoid log(0) in entropy calculations
EMD_MAX_ITER = 10000  # Maximum iterations for EMD calculation


def calculate_KL_EMD(
    dist1: np.ndarray,
    dist2: np.ndarray,
    reg_strength: float | None,
) -> tuple[float, float]:
    """
    Calculates the KL Divergence and EMD between two distributions of n variables.
    Can be used to calculate distance metrics for specific combinations of receptors.

    :param dist1: First distribution
    :param dist2: Second distribution
    :param reg_strength: Regularization strength for the EMD calculation. Also controls
        the OT solving algorithm used. If None or 0, the unregularized EMD is calculated
        using the Network Simplex algorithm with a block search pivot (Bonneel et al.
        2011). If a positive float, the KL divergence-regularized EMD is calculated
        using the Sinkhorn-Knopp matrix scaling algorithm (Cuturi 2013).
    :return:
        KL Divergence between the two distributions
        EMD between the two distributions

    In addition to the input parameters, there are a number of important parameters
    associated with the kernel density estimation and the EMD distance matrix
    calculation that can be adjusted.
    """

    assert dist1.shape[1] == dist2.shape[1]

    n_dim = dist1.shape[1]

    # Estimate the n-dimensional probability distributions
    kde1 = KernelDensity(atol=KDE_A_TOL, rtol=KDE_R_TOL, bandwidth=BW_METHOD).fit(dist1)
    kde2 = KernelDensity(atol=KDE_A_TOL, rtol=KDE_R_TOL, bandwidth=BW_METHOD).fit(dist2)

    # Compare over the entire distribution space by looking at the global max/min
    min_abun = np.minimum(dist1.min(axis=0), dist2.min(axis=0)) - KDE_GRID_MARGIN
    max_abun = np.maximum(dist1.max(axis=0), dist2.max(axis=0)) + KDE_GRID_MARGIN

    # Create a mesh grid for n-dimensional comparison
    grids = np.meshgrid(
        *[np.linspace(min_abun[i], max_abun[i], KDE_GRID_SIZE) for i in range(n_dim)]
    )
    grids = np.stack([grid.flatten() for grid in grids], axis=-1)

    # Calculate the probabilities (log-likelihood) of all combinations in the mesh grid
    dist1_probs = np.exp(kde1.score_samples(grids))
    dist2_probs = np.exp(kde2.score_samples(grids))

    # Calculate KL Divergence
    KL_div_val_1 = stats.entropy(
        dist2_probs + ENTROPY_EPS, dist1_probs + ENTROPY_EPS, base=2
    )
    KL_div_val_2 = stats.entropy(
        dist1_probs + ENTROPY_EPS, dist2_probs + ENTROPY_EPS, base=2
    )
    KL_div_val = float((KL_div_val_1 + KL_div_val_2) / 2)  # Symmetrized KL Divergence

    # Calculate Euclidean cost matrix
    M = ot.dist(dist1, dist2, metric="euclidean")

    # Solve the optimal transport problem
    EMD_res = ot.solve(M, reg=reg_strength, max_iter=EMD_MAX_ITER)

    # Extract the total transportation cost
    EMD_val = EMD_res.value_linear

    return KL_div_val, EMD_val


def KL_EMD_1D(
    recAbundances: np.ndarray,
    targ: np.ndarray,
    offTarg: np.ndarray,
    reg_strength: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates 1D EMD and KL Divergence between target and off-target populations.
    Calculates distance metrics for all receptors.

    :param recAbundances: abundances across cells (rows) and receptors (columns)
    :param targ: a numpy vector with boolean values indicating indices of target cells
    :param offTarg: vector with boolean values indicating indices of off-target cells
    :return:
        KL_div_vals: vector of KL Divergences per receptor
        EMD_vals: a vector of EMDs per receptor
    """

    assert all(
        isinstance(i, np.bool) for i in np.append(targ, offTarg)
    )  # Check that targ and offTarg are only boolean
    assert sum(targ) != 0 and sum(offTarg) != 0

    KL_div_vals = np.full(recAbundances.shape[1], np.nan)
    EMD_vals = np.full(recAbundances.shape[1], np.nan)

    targNorms = recAbundances[targ, :] / np.mean(recAbundances, axis=0)
    offTargNorms = recAbundances[offTarg, :] / np.mean(recAbundances, axis=0)

    assert targNorms.shape[0] == sum(
        targ
    )  # Check that only cells of target cell type are kept
    assert (
        targNorms.shape[0] != recAbundances.shape[0]
    )  # Check that not all cells are target cells

    for rec in range(recAbundances.shape[1]):
        targAbun = targNorms[:, rec]
        offTargAbun = offTargNorms[:, rec]

        assert np.allclose(
            targAbun, recAbundances[targ, rec] / np.mean(recAbundances[:, rec])
        )

        KL_div_vals[rec], EMD_vals[rec] = calculate_KL_EMD(
            targAbun.reshape(-1, 1),
            offTargAbun.reshape(-1, 1),
            reg_strength=reg_strength,
        )

    return KL_div_vals, EMD_vals


def KL_EMD_2D(
    recAbundances: np.ndarray,
    targ: np.ndarray,
    offTarg: np.ndarray,
    calc_1D: bool = True,
    reg_strength: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates 2D EMD and KL Divergence between the target and off-target populations
    of two receptors. Calculates distance metrics for all receptor pairs.

    :param recAbundances: abundances across cells (rows) and receptors (columns)
    :param targ: a numpy vector with boolean values indicating indices of target cells
    :param offTarg: vector with boolean values indicating indices of off-target cells
    :param calc_1D: whether to calculate 1D distances
    :return:
        KL_div_vals: an array of KL Divergences between the on- and off-target
            abundances of multiple receptors for each entry the value is calculated
            for two receptors, so every entry is for a different receptor pair the array
            is triangular because it is symmetric across the diagonal the diagonal
            values are the 1D distances
        EMD_vals: similar to KL_div_vals but with EMDs
    """

    assert all(isinstance(i, np.bool) for i in np.append(targ, offTarg))
    assert sum(targ) != 0 and sum(offTarg) != 0

    KL_div_vals = np.full((recAbundances.shape[1], recAbundances.shape[1]), np.nan)
    EMD_vals = np.full((recAbundances.shape[1], recAbundances.shape[1]), np.nan)

    targNorms = recAbundances[targ, :] / np.mean(recAbundances, axis=0)
    offTargNorms = recAbundances[offTarg, :] / np.mean(recAbundances, axis=0)

    assert targNorms.shape[0] == sum(targ)
    assert targNorms.shape[0] != recAbundances.shape[0]

    row, col = np.tril_indices(
        recAbundances.shape[1], k=0
    )  # k=0 includes diagonals which are 1D distances
    for rec1, rec2 in zip(row, col, strict=False):
        if not calc_1D and rec1 == rec2:
            continue

        targAbun1, targAbun2 = targNorms[:, rec1], targNorms[:, rec2]
        offTargAbun1, offTargAbun2 = offTargNorms[:, rec1], offTargNorms[:, rec2]

        assert np.allclose(
            targAbun1, recAbundances[targ, rec1] / np.mean(recAbundances[:, rec1])
        )

        targAbunAll = np.vstack((targAbun1, targAbun2)).transpose()
        offTargAbunAll = np.vstack((offTargAbun1, offTargAbun2)).transpose()

        assert targAbunAll.shape == (np.sum(targ), 2)

        KL_div_vals[rec1, rec2], EMD_vals[rec1, rec2] = calculate_KL_EMD(
            targAbunAll, offTargAbunAll, reg_strength=reg_strength
        )

    return KL_div_vals, EMD_vals


def KL_EMD_3D(
    recAbundances: np.ndarray,
    targ: np.ndarray,
    offTarg: np.ndarray,
    calc_diags: bool = True,
    reg_strength: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates 3D EMD and KL Divergence between the target and off-target populations
    of three receptors. Calculates distance metrics for all receptor triplets.

    :param recAbundances: abundances across cells (rows) and receptors (columns)
    :param targ: a numpy vector with boolean values indicating indices of target cells
    :param offTarg: vector with boolean values indicating indices of off-target cells
    :param calc_diags: whether to calculate 1D and 2D distances
    :return:
        KL_div_vals: an array of KL Divergences between the on- and off-target
            abundances of multiple receptors for each entry the value is calculated
            for three receptors, so every entry is for a different receptor triplet.
            The array is pyramidal because it is symmetric across the diagonal.
        EMD_vals: similar to KL_div_vals but with EMDs
    """

    assert all(isinstance(i, np.bool) for i in np.append(targ, offTarg))
    assert sum(targ) != 0 and sum(offTarg) != 0

    KL_div_vals = np.full(
        (recAbundances.shape[1], recAbundances.shape[1], recAbundances.shape[1]), np.nan
    )
    EMD_vals = np.full(
        (recAbundances.shape[1], recAbundances.shape[1], recAbundances.shape[1]), np.nan
    )

    targNorms = recAbundances[targ, :] / np.mean(recAbundances, axis=0)
    offTargNorms = recAbundances[offTarg, :] / np.mean(recAbundances, axis=0)

    assert targNorms.shape[0] == sum(targ)
    assert targNorms.shape[0] != recAbundances.shape[0]

    for rec1, rec2, rec3 in combinations_with_replacement(
        range(recAbundances.shape[1]), 3
    ):
        if not calc_diags and (rec1 in (rec2, rec3) or rec2 == rec3):
            continue

        targAbun1, targAbun2, targAbun3 = (
            targNorms[:, rec1],
            targNorms[:, rec2],
            targNorms[:, rec3],
        )
        offTargAbun1, offTargAbun2, offTargAbun3 = (
            offTargNorms[:, rec1],
            offTargNorms[:, rec2],
            offTargNorms[:, rec3],
        )

        assert np.allclose(
            targAbun1, recAbundances[targ, rec1] / np.mean(recAbundances[:, rec1])
        )

        targAbunAll = np.vstack((targAbun1, targAbun2, targAbun3)).transpose()
        offTargAbunAll = np.vstack(
            (offTargAbun1, offTargAbun2, offTargAbun3)
        ).transpose()

        assert targAbunAll.shape == (np.sum(targ), 3)

        KL_div_vals[rec1, rec2, rec3], EMD_vals[rec1, rec2, rec3] = calculate_KL_EMD(
            targAbunAll, offTargAbunAll, reg_strength=reg_strength
        )

    return KL_div_vals, EMD_vals
