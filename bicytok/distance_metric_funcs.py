import time
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import ot
import pandas as pd
from scipy import stats
from sklearn.neighbors import KernelDensity

from .imports import importCITE, sample_receptor_abundances

path_here = Path(__file__).parent.parent


def calculate_KL_EMD(dist1: np.ndarray, dist2: np.ndarray) -> tuple[float, float]:
    """
    Calculates the KL Divergence and EMD between two distributions of n variables.
    Can be used to calculate distance metrics for specific combinations of receptors.

    :param dist1: First distribution
    :param dist2: Second distribution
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
    start_kl = time.time()
    kde1 = KernelDensity(atol=1e-9, rtol=1e-9).fit(dist1)
    kde2 = KernelDensity(atol=1e-9, rtol=1e-9).fit(dist2)

    # Compare over the entire distribution space by looking at the global max/min
    min_abun = np.minimum(dist1.min(axis=0), dist2.min(axis=0)) - 100
    max_abun = np.maximum(dist1.max(axis=0), dist2.max(axis=0)) + 100

    # Create a mesh grid for n-dimensional comparison
    grids = np.meshgrid(
        *[np.linspace(min_abun[i], max_abun[i], 100) for i in range(n_dim)]
    )
    grids = np.stack([grid.flatten() for grid in grids], axis=-1)

    # Calculate the probabilities (log-likelihood) of all combinations in the mesh grid
    dist1_probs = np.exp(kde1.score_samples(grids))
    dist2_probs = np.exp(kde2.score_samples(grids))

    # Calculate KL Divergence
    KL_div_val = stats.entropy(dist2_probs + 1e-200, dist1_probs + 1e-200, base=2)
    print(f"KL Divergence calculation took {time.time() - start_kl} seconds")

    # Calculate Euclidean distance matrix
    start_emd = time.time()
    M = ot.dist(dist1, dist2, metric="euclidean")

    # Calculate EMD
    EMD_val = ot.emd2([], [], M)
    print(f"EMD calculation took {time.time() - start_emd} seconds")

    return KL_div_val, EMD_val


def KL_EMD_1D(
    recAbundances: np.ndarray, targ: np.ndarray, offTarg: np.ndarray
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

    # Filter indices based on the conditions
    valid_indices = [
        rec
        for rec in range(recAbundances.shape[1])
        if np.mean(recAbundances[:, rec]) > 5
        and np.mean(targNorms[:, rec]) > np.mean(offTargNorms[:, rec])
    ]

    for rec in valid_indices:
        targAbun = targNorms[:, rec]
        offTargAbun = offTargNorms[:, rec]

        assert np.allclose(
            targAbun, recAbundances[targ, rec] / np.mean(recAbundances[:, rec])
        )

        KL_div_vals[rec], EMD_vals[rec] = calculate_KL_EMD(
            targAbun.reshape(-1, 1), offTargAbun.reshape(-1, 1)
        )

    return KL_div_vals, EMD_vals


def KL_EMD_2D(
    recAbundances: np.ndarray,
    targ: np.ndarray,
    offTarg: np.ndarray,
    calc_1D: bool = True,
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

    # Filter indices based on the conditions
    valid_indices = [
        rec
        for rec in range(recAbundances.shape[1])
        if np.mean(recAbundances[:, rec]) > 5
        and np.mean(targNorms[:, rec]) > np.mean(offTargNorms[:, rec])
    ]

    row, col = np.tril_indices(
        recAbundances.shape[1], k=0
    )  # k=0 includes diagonals which are 1D distances
    for rec1, rec2 in zip(row, col, strict=False):
        if not calc_1D and rec1 == rec2:
            continue
        if rec1 not in valid_indices or rec2 not in valid_indices:
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
            targAbunAll, offTargAbunAll
        )

    return KL_div_vals, EMD_vals


def make_2D_distance_metrics():
    """
    Generates a CSV of 2D EMD and KL Divergence values for all receptors.
    Called with "rye run distanceCSV"
    """
    start = time.time()

    targCell = "Treg"
    receptors_of_interest = None  # Can be a list of receptors or None
    sample_size = 1000
    cell_categorization = "CellType2"
    cellTypes = np.array(
        [
            "CD8 Naive",
            "NK",
            "CD8 TEM",
            "CD4 Naive",
            "CD4 CTL",
            "CD8 TCM",
            "CD8 Proliferating",
            "Treg",
        ]
    )
    offTargCells = cellTypes[cellTypes != targCell]

    epitopesList = pd.read_csv(path_here / "bicytok" / "data" / "epitopeList.csv")
    epitopes = list(epitopesList["Epitope"].unique())
    CITE_DF = importCITE()

    # Randomly sample a subset of rows
    epitopesDF = CITE_DF[epitopes + [cell_categorization]]
    epitopesDF = epitopesDF.loc[epitopesDF[cell_categorization].isin(cellTypes)]
    epitopesDF = epitopesDF.rename(columns={cell_categorization: "Cell Type"})
    sampleDF = sample_receptor_abundances(
        CITE_DF=epitopesDF,
        numCells=min(sample_size, epitopesDF.shape[0]),
        targCellType=targCell,
        offTargCellTypes=offTargCells,
    )

    # Define target and off-target cell masks (for distance metrics)
    on_target_mask = (sampleDF["Cell Type"] == targCell).to_numpy()
    off_target_mask = sampleDF["Cell Type"].isin(offTargCells).to_numpy()

    if receptors_of_interest is not None:
        sampleDF = sampleDF[receptors_of_interest]
    else:
        sampleDF = sampleDF[epitopes]
    rec_abundances = sampleDF.to_numpy()
    receptors_of_interest = sampleDF.columns

    KL_div_vals, EMD_vals = KL_EMD_2D(
        rec_abundances, on_target_mask, off_target_mask, calc_1D=True
    )

    EMD_matrix = np.tril(EMD_vals, k=0)
    EMD_matrix = EMD_matrix + EMD_matrix.T - np.diag(np.diag(EMD_matrix))
    KL_matrix = np.tril(KL_div_vals, k=0)
    KL_matrix = KL_matrix + KL_matrix.T - np.diag(np.diag(KL_matrix))

    df_EMD = pd.DataFrame(
        EMD_matrix, index=receptors_of_interest, columns=receptors_of_interest
    )
    df_KL = pd.DataFrame(
        KL_matrix, index=receptors_of_interest, columns=receptors_of_interest
    )

    df_EMD.to_csv(
        path_here / "bicytok" / "data" / "2D_EMD_all.csv", header=True, index=True
    )
    df_KL.to_csv(
        path_here / "bicytok" / "data" / "2D_KL_div_all.csv", header=True, index=True
    )

    print(
        f"Completed generation of distance metric CSV in {time.time() - start} ",
        f"seconds on {time.ctime(time.time())}.",
    )


def KL_EMD_3D(
    recAbundances: np.ndarray,
    targ: np.ndarray,
    offTarg: np.ndarray,
    calc_diags: bool = True,
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

    # Filter indices based on the conditions
    valid_indices = [
        rec
        for rec in range(recAbundances.shape[1])
        if np.mean(recAbundances[:, rec]) > 5
        and np.mean(targNorms[:, rec]) > np.mean(offTargNorms[:, rec])
    ]

    for rec1, rec2, rec3 in combinations_with_replacement(valid_indices, 3):
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
            targAbunAll, offTargAbunAll
        )

    return KL_div_vals, EMD_vals


def test_runtimes(dim: int = 3, sample_size: int = 1000):
    """
    Tests the runtimes of a specified distance metric function.

    :param dim: the dimensionality of the distance metric function to test
    :param sample_size: the number of cells to sample from the dataset
    """

    assert dim in [1, 2, 3]

    targCell = "Treg"
    receptors_of_interest = ["CD25", "CD4-1", "CD27", "CD4-2"]
    cell_categorization = "CellType2"
    cellTypes = np.array(
        [
            "CD8 Naive",
            "NK",
            "CD8 TEM",
            "CD4 Naive",
            "CD4 CTL",
            "CD8 TCM",
            "CD8 Proliferating",
            "Treg",
        ]
    )
    offTargCells = cellTypes[cellTypes != targCell]

    epitopesList = pd.read_csv(path_here / "bicytok" / "data" / "epitopeList.csv")
    epitopes = list(epitopesList["Epitope"].unique())
    CITE_DF = importCITE()

    # Randomly sample a subset of rows
    epitopesDF = CITE_DF[epitopes + [cell_categorization]]
    epitopesDF = epitopesDF.loc[epitopesDF[cell_categorization].isin(cellTypes)]
    epitopesDF = epitopesDF.rename(columns={cell_categorization: "Cell Type"})
    sampleDF = sample_receptor_abundances(
        CITE_DF=epitopesDF,
        numCells=min(sample_size, epitopesDF.shape[0]),
        targCellType=targCell,
        offTargCellTypes=offTargCells,
    )

    # Define target and off-target cell masks (for distance metrics)
    on_target_mask = (sampleDF["Cell Type"] == targCell).to_numpy()
    off_target_mask = sampleDF["Cell Type"].isin(offTargCells).to_numpy()

    if receptors_of_interest is not None:
        sampleDF = sampleDF[receptors_of_interest]
    else:
        sampleDF = sampleDF[epitopes]
    rec_abundances = sampleDF.to_numpy()
    receptors_of_interest = sampleDF.columns

    start = time.time()

    if dim == 1:
        KL_div_vals, EMD_vals = KL_EMD_1D(
            rec_abundances,
            on_target_mask,
            off_target_mask,
        )
    elif dim == 2:
        KL_div_vals, EMD_vals = KL_EMD_2D(
            rec_abundances, on_target_mask, off_target_mask, calc_1D=False
        )
    elif dim == 3:
        KL_div_vals, EMD_vals = KL_EMD_3D(
            rec_abundances, on_target_mask, off_target_mask, calc_diags=False
        )

    print(f"Completed in {time.time() - start} seconds on {time.ctime(time.time())}.")

    print(KL_div_vals)
    print(EMD_vals)
