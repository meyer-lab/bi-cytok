import numpy as np
import ot
from scipy import stats
from sklearn.neighbors import KernelDensity
from itertools import combinations_with_replacement


def calculateKLDiv(
    targAbun: np.ndarray, offTargAbun: np.ndarray, dims: np.ndarray
) -> float:
    """
    Calculates KL Divergence between target and off-target abundance distributions
    :param targAbun: in 1D - a vector of target abundance values for one receptor
        in 2D - an nx2 array where each column are the target abundance values for one receptor
    :param offTargAbun: in 1D - a vector of off-target abundance values for one receptor
        in 2D - an nx2 array where each column are the off-target abundance values for one receptor
    :param dims: the number of receptors being compared across
    :return:
        KL_div_val: the KL Divergence between target and off-target abundances
    """

    if dims == 1:
        targKDE = KernelDensity(kernel="gaussian").fit(targAbun.reshape(-1, 1))
        offTargKDE = KernelDensity(kernel="gaussian").fit(offTargAbun.reshape(-1, 1))
        minAbun = np.minimum(targAbun.min(), offTargAbun.min()) - 10
        maxAbun = np.maximum(targAbun.max(), offTargAbun.max()) + 10
        outcomes = np.arange(minAbun, maxAbun + 1).reshape(-1, 1)

    elif dims == 2:  # LOOK FOR WAY TO GENERALIZE FOR ALL DIMS
        targKDE = KernelDensity(kernel="gaussian").fit(targAbun.reshape(-1, 2))
        offTargKDE = KernelDensity(kernel="gaussian").fit(offTargAbun.reshape(-1, 2))
        minAbun = np.minimum(targAbun.min(), offTargAbun.min()) - 10
        maxAbun = np.maximum(targAbun.max(), offTargAbun.max()) + 10
        X, Y = np.mgrid[
            minAbun : maxAbun : ((maxAbun - minAbun) / 100),
            minAbun : maxAbun : ((maxAbun - minAbun) / 100),
        ]
        outcomes = np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1)

    elif dims == 3:
        raise NotImplementedError("3D KL Divergence not yet implemented")

    targDist = np.exp(targKDE.score_samples(outcomes))
    offTargDist = np.exp(offTargKDE.score_samples(outcomes))
    KL_div_val = stats.entropy(
        offTargDist.flatten() + 1e-200, targDist.flatten() + 1e-200, base=2
    )

    return KL_div_val


def KL_EMD_1D(
    recAbundances: np.ndarray, targ: np.ndarray, offTarg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates 1D EMD and KL Divergence between target and off-target populations within multiple receptors
    :param recAbundances: an array of receptor abundances across cells (rows) and receptors (columns)
    :param targ: a binary/boolian vector indicating indices of target cells
    :param offTarg: a binary/boolian vector indicating indices of off-target cells
    :return:
        KL_div_vals: a vector of KL Divergences where each entry is the value for one receptor
        EMD_vals: a vector of EMDs where each entry is the value for one receptor
    """

    KL_div_vals = np.full(recAbundances.shape[1], np.nan)
    EMD_vals = np.full(recAbundances.shape[1], np.nan)

    targNorms = recAbundances[targ, :] / np.mean(recAbundances[targ, :], axis=0)
    offTargNorms = recAbundances[offTarg, :] / np.mean(recAbundances[offTarg, :], axis=0)

    for rec in range(recAbundances.shape[1]):
        if (  # Check these with pre-normalization values
            np.mean(recAbundances[:, rec]) > 5
            and np.mean(recAbundances[targ, rec]) > np.mean(recAbundances[offTarg, rec])
        ):
            targAbun = targNorms[:, rec]
            offTargAbun = offTargNorms[:, rec]

            assert targAbun == recAbundances[targ, rec] / np.mean(recAbundances[targ, rec])

            KL_div_vals[rec] = calculateKLDiv(targAbun, offTargAbun, dims=1)
            EMD_vals[rec] = stats.wasserstein_distance(
                targAbun, offTargAbun
            )  # Consider switching/comparing to ot.emd2_1d

    return KL_div_vals, EMD_vals


def KL_EMD_2D(
    recAbundances: np.ndarray, targ: np.ndarray, offTarg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates 2D EMD and KL Divergence between the target and off-target populations of two receptors, across multiple receptors
    :param recAbundances: an array of receptor abundances across cells (rows) and receptors (columns)
    :param targ: a binary/boolian vector indicating indices of target cells
    :param offTarg: a binary/boolian vector indicating indices of off-target cells
    :return:
        KL_div_vals: an array of KL Divergences between the on- and off-target abundances of multiple receptors
            for each entry the value is calculated for two receptors, so every entry is for a different receptor pair
            the array is triangular because it is symmetric across the diagonal
            the diagonal values are the 1D distances
        EMD_vals: similar to KL_div_vals but with EMDs
    """

    KL_div_vals = np.full((recAbundances.shape[1], recAbundances.shape[1]), np.nan)
    EMD_vals = np.full((recAbundances.shape[1], recAbundances.shape[1]), np.nan)

    targNorms = recAbundances[targ, :] / np.mean(recAbundances[targ, :], axis=0)
    offTargNorms = recAbundances[offTarg, :] / np.mean(recAbundances[offTarg, :], axis=0)

    row, col = np.tril_indices(recAbundances.shape[1])  # Triangle indices, includes diagonal (k=0 by default)
    for rec1, rec2 in zip(row, col):
        if (
            np.mean(recAbundances[:, rec1]) > 5
            and np.mean(recAbundances[:, rec2]) > 5
            and np.mean(recAbundances[targ, rec1]) > np.mean(recAbundances[offTarg, rec1])
            and np.mean(recAbundances[targ, rec2]) > np.mean(recAbundances[offTarg, rec2])
        ):
            targAbun1, targAbun2 = targNorms[:, rec1], targNorms[:, rec2]
            offTargAbun1, offTargAbun2 = offTargNorms[:, rec1], offTargNorms[:, rec2]

            assert targAbun1 == recAbundances[targ, rec1] / np.mean(recAbundances[targ, rec1])

            targAbunAll = np.vstack((targAbun1, targAbun2)).transpose()
            offTargAbunAll = np.vstack((offTargAbun1, offTargAbun2)).transpose()

            assert targAbunAll.shape == (np.sum(targ), 2)

            KL_div_vals[rec1, rec2] = calculateKLDiv(
                targAbunAll, offTargAbunAll, dims=2
            )

            M = ot.dist(targAbunAll, offTargAbunAll)
            a = np.ones((targAbunAll.shape[0],)) / targAbunAll.shape[0]
            b = np.ones((offTargAbunAll.shape[0],)) / offTargAbunAll.shape[0]
            EMD_vals[rec1, rec2] = ot.emd2(
                a, b, M, numItermax=100  # Check numIterMax
            )

    return KL_div_vals, EMD_vals


def KL_EMD_3D(
    recAbundances: np.ndarray, targ: np.ndarray, offTarg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    3D applicatin of KL Divergence and EMD
    NOT YET IMPLEMENTED
    """

    KL_div_vals = np.full((recAbundances.shape[1], recAbundances.shape[1], recAbundances.shape[1]), np.nan)
    EMD_vals = np.full((recAbundances.shape[1], recAbundances.shape[1], recAbundances.shape[1]), np.nan)

    targNorms = recAbundances[targ, :] / np.mean(recAbundances[targ, :], axis=0)
    offTargNorms = recAbundances[offTarg, :] / np.mean(recAbundances[offTarg, :], axis=0)

    for rec1, rec2, rec3 in combinations_with_replacement(range(recAbundances.shape[1]), 3):  # 3D triangle (pyramidal?) indices, with replacement includes diagonals
        if (
            np.mean(recAbundances[:, rec1]) > 5
            and np.mean(recAbundances[:, rec2]) > 5
            and np.mean(recAbundances[:, rec3]) > 5
            and np.mean(recAbundances[targ, rec1]) > np.mean(recAbundances[offTarg, rec1])
            and np.mean(recAbundances[targ, rec2]) > np.mean(recAbundances[offTarg, rec2])
            and np.mean(recAbundances[targ, rec3]) > np.mean(recAbundances[offTarg, rec3])
        ):
            targAbun1, targAbun2, targAbun3 = targNorms[:, rec1], targNorms[:, rec2], targNorms[:, rec3]
            offTargAbun1, offTargAbun2, offTargAbun3 = offTargNorms[:, rec1], offTargNorms[:, rec2], offTargNorms[:, rec3]

            assert targAbun1 == recAbundances[targ, rec1] / np.mean(recAbundances[targ, rec1])

            targAbunAll = np.vstack(
                (targAbun1, targAbun2, targAbun3)
            ).transpose()
            offTargAbunAll = np.vstack(
                (offTargAbun1, offTargAbun2, offTargAbun3)
            ).tranpose()  # Check that this is the same as original concat

            assert targAbunAll.shape == (np.sum(targ), 3)

            KL_div_vals[rec1, rec2, rec3] = calculateKLDiv(
                targAbunAll, offTargAbunAll, dims=3
            )

            M = ot.dist(targAbunAll, offTargAbunAll)
            a = np.ones((targAbunAll.shape[0],)) / targAbunAll.shape[0]
            b = np.ones((offTargAbunAll.shape[0],)) / offTargAbunAll.shape[0]
            EMD_vals[rec1, rec2, rec3] = ot.emd2(
                a, b, M, numItermax=100  # Check numIterMax
            ) 

    return KL_div_vals, EMD_vals
