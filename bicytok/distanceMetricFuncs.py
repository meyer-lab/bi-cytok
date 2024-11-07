import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy import stats
import ot
import ot.plot
from .selectivityFuncs import convFactCalc, getSampleAbundances
from os.path import dirname
from .imports import importCITE


def normalize(recAbun, targ, offTarg):
    """
    Normalizes the target and off-target abundance values within one receptor
    :param recAbun: a vector of CITE-seq abundance values
    :param targ: a binary/boolian vector indicating indices of target cells
    :param offTarg: a binary/boolian vector indicating indices of off-target cells
    :return:
        targAbun: a vector of normalized abundance values for all target cells
        offTargAbun: a vector of normalized abundance values for all off-target cells
    """

    recAvg = np.mean(recAbun)

    assert recAvg != 0

    targAbun = recAbun[targ] / recAvg
    offTargAbun = recAbun[offTarg] / recAvg

    return targAbun, offTargAbun


def calculateKLDiv(targAbun, offTargAbun, dims):
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

    targDist = np.exp(targKDE.score_samples(outcomes))
    offTargDist = np.exp(offTargKDE.score_samples(outcomes))
    KL_div_val = stats.entropy(
        offTargDist.flatten() + 1e-200, targDist.flatten() + 1e-200, base=2
    )

    return KL_div_val


def KL_EMD_1D(recAbundances, targ, offTarg):
    """
    Calculates 1D EMD and KL Divergence between target and off-target populations within multiple receptors
    :param recAbundances: an array of receptor abundances across cells (rows) and receptors (columns)
    :param targ: a binary/boolian vector indicating indices of target cells
    :param offTarg: a binary/boolian vector indicating indices of off-target cells
    :return:
        KL_div_vals: a vector of KL Divergences where each entry is the value for one receptor
        EMD_vals: a vector of EMDs where each entry is the value for one receptor
    """

    KL_div_vals = np.empty(recAbundances.shape[1])
    EMD_vals = np.empty(recAbundances.shape[1])

    for rec in range(recAbundances.shape[1]):
        targAbun, offTargAbun = normalize(recAbundances[:, rec], targ, offTarg)

        if np.mean(targAbun) > np.mean(offTargAbun):
            KL_div_vals[rec] = calculateKLDiv(targAbun, offTargAbun, dims=1)
            EMD_vals[rec] = stats.wasserstein_distance(
                targAbun, offTargAbun
            )  # Consider switching/comparing to ot.emd2_1d
        else:
            KL_div_vals[rec] = np.nan
            EMD_vals[rec] = np.nan

    return KL_div_vals, EMD_vals


def KL_EMD_2D(recAbundances, targ, offTarg):
    """
    Calculates 2D EMD and KL Divergence between the target and off-target populations of two receptors, across multiple receptors
    :param recAbundances: an array of receptor abundances across cells (rows) and receptors (columns)
    :param targ: a binary/boolian vector indicating indices of target cells
    :param offTarg: a binary/boolian vector indicating indices of off-target cells
    :return:
        KL_div_vals: an array of KL Divergences between the on- and off-target abundances of multiple receptors
            for each entry the value is calculated for two receptors, so every entry is for a different receptor pair
            the array is triangular because it is symmetric across the diagonal
            ???the diagonal values are the 1D distances??? Check this
        EMD_vals: similar to KL_div_vals but with EMDs
    """

    # Do conversion factors in functions?
    # weightDF = convFactCalc(recAbundances) ...

    KL_div_vals = np.empty([recAbundances.shape[1], recAbundances.shape[1]])
    KL_div_vals[:] = np.nan
    EMD_vals = np.empty([recAbundances.shape[1], recAbundances.shape[1]])
    EMD_vals[:] = np.nan

    for rec1 in range(recAbundances.shape[1]):
        targAbun1, offTargAbun1 = normalize(recAbundances[:, rec1], targ, offTarg)

        # for rec2 in range(recAbundances.shape[1]): #Make square matrix outputs (symmetrical across diagonal)
        for rec2 in range(rec1 + 1):  # Make triangular matrix outputs
            targAbun2, offTargAbun2 = normalize(recAbundances[:, rec2], targ, offTarg)

            # How to interpret case when rec1==rec2?
            # -Could result in 1D distances
            # -If so, can avoid running those cases for speed.
            # if rec1 == rec2:
            #     EMD_vals[rec2, rec1] = 0
            #     KL_div_vals[rec2, rec1] = 0

            if (
                np.mean(recAbundances[:, rec1]) > 5
                and np.mean(recAbundances[:, rec2]) > 5
                and np.mean(targAbun1) > np.mean(offTargAbun1)
                and np.mean(targAbun2) > np.mean(offTargAbun2)
            ):
                targAbunAll = np.vstack((targAbun1, targAbun2)).transpose()
                offTargAbunAll = np.vstack((offTargAbun1, offTargAbun2)).transpose()

                assert targAbunAll.shape == (np.sum(targ), 2)

                KL_div_vals[rec2, rec1] = calculateKLDiv(
                    targAbunAll, offTargAbunAll, dims=2
                )

                M = ot.dist(targAbunAll, offTargAbunAll)
                a = np.ones((targAbunAll.shape[0],)) / targAbunAll.shape[0]
                b = np.ones((offTargAbunAll.shape[0],)) / offTargAbunAll.shape[0]
                EMD_vals[rec2, rec1] = ot.emd2(
                    a, b, M, numItermax=1
                )  # Check numIterMax

            else:
                KL_div_vals[rec2, rec1] = None
                EMD_vals[rec2, rec1] = None

    return KL_div_vals, EMD_vals


def KL_EMD_3D(recAbundances, targ, offTarg):
    """
    3D applicatin of KL Divergence and EMD
    NOT YET IMPLEMENTED
    """

    x = recAbundances

    KL_div_vals = np.empty([x.shape[1], x.shape[1], x.shape[1]])
    KL_div_vals[:] = np.nan
    EMD_vals = np.empty([x.shape[1], x.shape[1], x.shape[1]])
    EMD_vals[:] = np.nan

    for rec1 in range(x.shape[1]):
        targAbun1, offTargAbun1 = normalize(x[:, rec1], targ, offTarg)

        for rec2 in range(rec1 + 1):
            targAbun2, offTargAbun2 = normalize(x[:, rec2], targ, offTarg)

            for rec3 in range(
                rec2 + 1
            ):  # Check that this correctly makes "pyramidal" output matrix
                targAbun3, offTargAbun3 = normalize(x[:, rec3], targ, offTarg)

                # if rec1 == rec2 and rec1 == rec3 and rec2 == rec3: #Once again, avoid calculating 1D distances?
                #     EMD_vals[rec3, rec2, rec1] = 0
                #     KL_div_val[rec3, rec2, rec1] = 0
                # Could avoid 2D distances with from rec1 == rec2 and rec1, rec2 != rec3?

                if (
                    np.mean(x[:, rec1]) > 5
                    and np.mean(x[:, rec2]) > 5
                    and np.mean(x[:, rec3]) > 5
                    and np.mean(targAbun1) > np.mean(offTargAbun1)
                    and np.mean(targAbun2) > np.mean(offTargAbun2)
                    and np.mean(targAbun3) > np.mean(offTargAbun3)
                ):
                    targAbunAll = np.vstack(
                        (targAbun1, targAbun2, targAbun3)
                    ).transpose()
                    offTargAbunAll = np.vstack(
                        (offTargAbun1, offTargAbun2, offTargAbun3)
                    ).tranpose()  # check that this is the same as original concat

                    assert targAbunAll.shape == (np.sum(targ), 3)

                    KL_div_vals[rec3, rec2, rec1] = calculateKLDiv(
                        targAbunAll, offTargAbunAll, dims=3
                    )

                    M = ot.dist(targAbunAll, offTargAbunAll)
                    a = np.ones((targAbunAll.shape[0],)) / targAbunAll.shape[0]
                    b = np.ones((offTargAbunAll.shape[0],)) / offTargAbunAll.shape[0]
                    EMD_vals[rec3, rec2, rec1] = ot.emd2(
                        a, b, M, numItermax=100
                    )  # Check numIterMax

                else:
                    KL_div_vals[rec3, rec2, rec1] = None
                    EMD_vals[rec3, rec2, rec1] = None

    return KL_div_vals, EMD_vals
