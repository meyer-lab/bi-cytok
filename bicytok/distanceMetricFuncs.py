from itertools import combinations_with_replacement

import numpy as np
import ot
from scipy import stats
from sklearn.neighbors import KernelDensity


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

        assert np.allclose(
            targAbun, recAbundances[targ, rec] / np.mean(recAbundances[:, rec])
        )

        targKDE = KernelDensity(kernel="gaussian").fit(targAbun.reshape(-1, 1))
        offTargKDE = KernelDensity(kernel="gaussian").fit(
            offTargAbun.reshape(-1, 1)
        )
        minAbun = np.minimum(targAbun.min(), offTargAbun.min()) - 10
        maxAbun = np.maximum(targAbun.max(), offTargAbun.max()) + 10
        outcomes = np.arange(minAbun, maxAbun + 1).reshape(-1, 1)
        targDist = np.exp(targKDE.score_samples(outcomes))
        offTargDist = np.exp(offTargKDE.score_samples(outcomes))
        KL_div_vals[rec] = stats.entropy(
            offTargDist.flatten() + 1e-200, targDist.flatten() + 1e-200, base=2
        )

        EMD_vals[rec] = stats.wasserstein_distance(
            targAbun, offTargAbun
        )  # Consider switching/comparing to ot.emd2_1d

    return KL_div_vals, EMD_vals


def KL_EMD_2D(
    recAbundances: np.ndarray, targ: np.ndarray, offTarg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates 2D EMD and KL Divergence between the target and off-target populations
    of two receptors, across multiple receptors

    :param recAbundances: abundances across cells (rows) and receptors (columns)
    :param targ: a numpy vector with boolean values indicating indices of target cells
    :param offTarg: vector with boolean values indicating indices of off-target cells
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
        recAbundances.shape[1]
    )  # Triangle indices, includes diagonal (k=0 by default)
    for rec1, rec2 in zip(row, col, strict=False):
        targAbun1, targAbun2 = targNorms[:, rec1], targNorms[:, rec2]
        offTargAbun1, offTargAbun2 = offTargNorms[:, rec1], offTargNorms[:, rec2]

        assert np.allclose(
            targAbun1, recAbundances[targ, rec1] / np.mean(recAbundances[:, rec1])
        )

        targAbunAll = np.vstack((targAbun1, targAbun2)).transpose()
        offTargAbunAll = np.vstack((offTargAbun1, offTargAbun2)).transpose()

        assert targAbunAll.shape == (np.sum(targ), 2)

        targKDE = KernelDensity(kernel="gaussian").fit(targAbunAll.reshape(-1, 2))
        offTargKDE = KernelDensity(kernel="gaussian").fit(
            offTargAbunAll.reshape(-1, 2)
        )
        minAbun = np.minimum(targAbunAll.min(), offTargAbunAll.min()) - 10
        maxAbun = np.maximum(targAbunAll.max(), offTargAbunAll.max()) + 10
        X, Y = np.mgrid[
            minAbun : maxAbun : ((maxAbun - minAbun) / 100),
            minAbun : maxAbun : ((maxAbun - minAbun) / 100),
        ]
        outcomes = np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1)
        targDist = np.exp(targKDE.score_samples(outcomes))
        offTargDist = np.exp(offTargKDE.score_samples(outcomes))
        KL_div_vals[rec1, rec2] = stats.entropy(
            offTargDist.flatten() + 1e-200, targDist.flatten() + 1e-200, base=2
        )

        M = ot.dist(targAbunAll, offTargAbunAll)
        a = np.ones((targAbunAll.shape[0],)) / targAbunAll.shape[0]
        b = np.ones((offTargAbunAll.shape[0],)) / offTargAbunAll.shape[0]
        EMD_vals[rec1, rec2] = ot.emd2(
            a,
            b,
            M,
            numItermax=100,  # Check numIterMax
        )

    return KL_div_vals, EMD_vals


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

        M = ot.dist(targAbunAll, offTargAbunAll)
        a = np.ones((targAbunAll.shape[0],)) / targAbunAll.shape[0]
        b = np.ones((offTargAbunAll.shape[0],)) / offTargAbunAll.shape[0]
        EMD_vals[rec1, rec2, rec3] = ot.emd2(
            a,
            b,
            M,
            numItermax=100,  # Check numIterMax
        )

    return KL_div_vals, EMD_vals
