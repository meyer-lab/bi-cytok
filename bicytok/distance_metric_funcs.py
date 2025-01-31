from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import KernelDensity

from bicytok.imports import importCITE

path_here = Path(__file__).parent.parent


def KL_EMD_1D(
    recAbundances: np.ndarray, targ: np.ndarray, offTarg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates 1D EMD and KL Divergence between target and off-target populations within
    multiple receptors

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

        # Filters remove edge cases that effect KL div, but not EMD
        # First filter removes receptors that have low overall expression. Even if
        #   such receptors were differentially expressed, they would not be useful?
        # Second filter removes receptors that are differentially expressed BUT have low
        #   expression on target cells and thus cannot be used for selective targeting
        #   eg IgE, CD272
        if np.mean(recAbundances[:, rec]) > 5 and np.mean(targAbun) > np.mean(
            offTargAbun
        ):
            assert np.allclose(
                targAbun, recAbundances[targ, rec] / np.mean(recAbundances[:, rec])
            )

            targKDE = KernelDensity(kernel="gaussian").fit(targAbun.reshape(-1, 1))
            offTargKDE = KernelDensity(kernel="gaussian").fit(
                offTargAbun.reshape(-1, 1)
            )
            minAbun = np.minimum(targAbun.min(), offTargAbun.min()) - 100
            maxAbun = np.maximum(targAbun.max(), offTargAbun.max()) + 100
            X = np.mgrid[minAbun : maxAbun : (maxAbun - minAbun) / 100]
            outcomes = X.reshape(-1, 1)
            targDist = np.exp(targKDE.score_samples(outcomes))
            offTargDist = np.exp(offTargKDE.score_samples(outcomes))
            KL_div_vals[rec] = stats.entropy(
                offTargDist.flatten() + 1e-200, targDist.flatten() + 1e-200, base=2
            )

            EMD_vals[rec] = stats.wasserstein_distance(targAbun, offTargAbun)

    return KL_div_vals, EMD_vals


def KL_EMD_2D(
    recAbundances: np.ndarray,
    targ: np.ndarray,
    offTarg: np.ndarray,
    calc_1D: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates 2D EMD and KL Divergence between the target and off-target populations
    of two receptors, across multiple receptors

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

        if (
            np.mean(recAbundances[:, rec1]) > 5
            and np.mean(recAbundances[:, rec2]) > 5
            and np.mean(targAbun1) > np.mean(offTargAbun1)
            and np.mean(targAbun2) > np.mean(offTargAbun2)
        ):
            assert np.allclose(
                targAbun1, recAbundances[targ, rec1] / np.mean(recAbundances[:, rec1])
            )

            targAbunAll = np.vstack((targAbun1, targAbun2)).transpose()
            offTargAbunAll = np.vstack((offTargAbun1, offTargAbun2)).transpose()

            assert targAbunAll.shape == (np.sum(targ), 2)

            # Estimate the 2D probability distributions of the two receptors
            targKDE = KernelDensity(kernel="gaussian").fit(targAbunAll)
            offTargKDE = KernelDensity(kernel="gaussian").fit(offTargAbunAll)

            # Compare over the entire distribution space by looking at the
            #   global max/min
            minAbun = np.minimum(targAbunAll.min(), offTargAbunAll.min()) - 100
            maxAbun = np.maximum(targAbunAll.max(), offTargAbunAll.max()) + 100

            # Need a mesh grid for 2D comparison because
            #   need to explore the entire distribution space
            X, Y = np.mgrid[
                minAbun : maxAbun : ((maxAbun - minAbun) / 100),
                minAbun : maxAbun : ((maxAbun - minAbun) / 100),
            ]
            outcomes = np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1)

            # Calculate the probabilities (log-likelihood)
            #   of all X, Y combos in the mesh grid based on the estimates
            #   of the two receptor distributions
            targDist = np.exp(targKDE.score_samples(outcomes))
            offTargDist = np.exp(offTargKDE.score_samples(outcomes))

            KL_div_vals[rec1, rec2] = stats.entropy(
                offTargDist + 1e-200, targDist + 1e-200, base=2
            )

            EMD_vals[rec1, rec2] = stats.wasserstein_distance_nd(
                targAbunAll, offTargAbunAll
            )

    return KL_div_vals, EMD_vals


def make_2D_distance_metrics():
    targCell = "Treg"
    offTargState = 1
    receptors_of_interest = [
        "CD25",
        "CD4-1",
        "CD27",
        "CD4-2",
        "CD278",
        "CD28",
        "CD45RB",
    ]
    sample_size = 100

    assert any(np.array([0, 1, 2]) == offTargState)

    CITE_DF = importCITE()

    # Define non-marker columns
    non_marker_columns = ["CellType1", "CellType2", "CellType3", "Cell"]
    marker_columns = CITE_DF.columns[~CITE_DF.columns.isin(non_marker_columns)]
    markerDF = CITE_DF.loc[:, marker_columns]

    if receptors_of_interest is not None:
        filtered_markerDF = markerDF.loc[
            :,
            markerDF.columns.str.fullmatch("|".join(receptors_of_interest), case=False),
        ]
    else:
        filtered_markerDF = markerDF
    receptors_of_interest = filtered_markerDF.columns

    on_target = (CITE_DF["CellType2"] == targCell).to_numpy()
    off_target_conditions = {
        0: (CITE_DF["CellType3"] != targCell),  # All non-memory Tregs
        1: (
            (CITE_DF["CellType2"] != "Treg") & (CITE_DF["CellType2"] != targCell)
        ),  # All non-Tregs
        2: (CITE_DF["CellType3"] == "Treg Naive"),  # Naive Tregs
    }
    off_target = off_target_conditions[offTargState].to_numpy()

    rec_abundances = filtered_markerDF.to_numpy()

    # Randomly sample a subset of rows
    if sample_size is not None:
        subset_indices = np.random.choice(
            len(on_target), size=min(sample_size, len(on_target)), replace=False
        )
        on_target = on_target[subset_indices]
        off_target = off_target[subset_indices]
        rec_abundances = rec_abundances[subset_indices]

    KL_div_vals, EMD_vals = KL_EMD_2D(
        rec_abundances, on_target, off_target, calc_1D=True
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


def KL_EMD_3D(
    recAbundances: np.ndarray, targ: np.ndarray, offTarg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    3D applicatin of KL Divergence and EMD
    NOT YET IMPLEMENTED
    """

    raise NotImplementedError("3D KL Divergence and EMD not yet implemented")

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
    ):  # 3D triangle (pyramidal?) indices, with replacement includes diagonals
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

        assert all(
            targAbun1 == recAbundances[targ, rec1] / np.mean(recAbundances[:, rec1])
        )

        targAbunAll = np.vstack((targAbun1, targAbun2, targAbun3)).transpose()
        offTargAbunAll = np.vstack(
            (offTargAbun1, offTargAbun2, offTargAbun3)
        ).tranpose()  # Check that this is the same as original concat

        assert targAbunAll.shape == (np.sum(targ), 3)

        KL_div_vals[rec1, rec2, rec3] = 0  # Placeholder unimplemented

        EMD_vals[rec1, rec2, rec3] = stats.wasserstein_distance_nd(
            targAbunAll, offTargAbunAll
        )

    return KL_div_vals, EMD_vals
