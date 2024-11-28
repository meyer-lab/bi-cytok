<<<<<<< HEAD
from os.path import dirname

import numpy as np
import ot
import ot.plot
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.neighbors import KernelDensity

from .imports import importCITE
from .selectivityFuncs import calcReceptorAbundances, convFactCalc

path_here = dirname(dirname(__file__))


# figure 2
def KL_EMD_1D(ax, targCell, numFactors, offTargState=0) -> pd.DataFrame:
    """
    Finds markers which have average greatest difference (EMD and KL) from other cells
    :param ax: Axes to plot on
    :param targCell: Target cell type for analysis
    :param numFactors: Number of top factors to consider
    Armaan: offTargState values should be defined as constant integers or
    strings so that this code is more readable.
    :param offTargState: State of off-target comparison (0 for all non-memory Tregs,
        1 for all non-Tregs, 2 for naive Tregs)
    :return:
        corrsDF: DataFrame containing marker information and their Wasserstein Distance
            and KL Divergence values
    """
    CITE_DF = importCITE()
    markerDF = pd.DataFrame(columns=["Marker", "Cell Type", "Amount"])
    for marker in CITE_DF.loc[
        :,
        (
            # Armaan: I think you can use df.isin here
            (CITE_DF.columns != "CellType1")
            & (CITE_DF.columns != "CellType2")
            & (CITE_DF.columns != "CellType3")
            & (CITE_DF.columns != "Cell")
        ),
    ].columns:
        markAvg = np.mean(CITE_DF[marker].values)
        if markAvg > 0.0001:
            targCellMark = (
                CITE_DF.loc[CITE_DF["CellType3"] == targCell][marker].values / markAvg
            )
            # Armaan: isn't this comment outdated? I think it's comparing to all
            # cell types that are not targCell
            # Compare to all non-memory Tregs
            if offTargState == 0:
                offTargCellMark = (
                    CITE_DF.loc[CITE_DF["CellType3"] != targCell][marker].values
                    / markAvg
                )
            # Armaan: what if the targcell is in the not Treg category?
            # Compare to all non-Tregs
            elif offTargState == 1:
                offTargCellMark = (
                    CITE_DF.loc[CITE_DF["CellType2"] != "Treg"][marker].values / markAvg
                )
            # Armaan: what if the targcell is Treg Naive?
            # Compare to naive Tregs
            elif offTargState == 2:
                offTargCellMark = (
                    CITE_DF.loc[CITE_DF["CellType3"] == "Treg Naive"][marker].values
                    / markAvg
                )
            if np.mean(targCellMark) > np.mean(offTargCellMark):
                kdeTarg = KernelDensity(kernel="gaussian").fit(
                    targCellMark.reshape(-1, 1)
                )
                kdeOffTarg = KernelDensity(kernel="gaussian").fit(
                    offTargCellMark.reshape(-1, 1)
                )
                minVal = np.minimum(targCellMark.min(), offTargCellMark.min()) - 10
                maxVal = np.maximum(targCellMark.max(), offTargCellMark.max()) + 10
                outcomes = np.arange(minVal, maxVal + 1).reshape(-1, 1)
                distTarg = np.exp(kdeTarg.score_samples(outcomes))
                distOffTarg = np.exp(kdeOffTarg.score_samples(outcomes))
                KL_div = stats.entropy(
                    distOffTarg.flatten() + 1e-200, distTarg.flatten() + 1e-200, base=2
                )
                markerDF = pd.concat(
                    [
                        markerDF,
                        pd.DataFrame(
                            {
                                "Marker": [marker],
                                "Wasserstein Distance": stats.wasserstein_distance(
                                    targCellMark, offTargCellMark
                                ),
                                "KL Divergence": KL_div,
                            }
                        ),
                    ]
                )

    corrsDF = pd.DataFrame()

    for i, distance in enumerate(["Wasserstein Distance", "KL Divergence"]):
        ratioDF = markerDF.sort_values(by=distance)
        posCorrs = ratioDF.tail(numFactors).Marker.values
        corrsDF = pd.concat(
            [corrsDF, pd.DataFrame({"Distance": distance, "Marker": posCorrs})]
        )
        # Armaan: I'm not sure why you are filtering markerDF here. Aren't you
        # plotting the top numFactors markers here independently for each
        # distance metric? Additionally, since the return value of this function
        # isn't used anywhere, you can get rid of all of the corrsDF stuff,
        # calculate ratioDF once, and plot it.
        markerDF = markerDF.loc[markerDF["Marker"].isin(posCorrs)]
        sns.barplot(
            data=ratioDF.tail(numFactors), y="Marker", x=distance, ax=ax[i], color="k"
        )
        ax[i].set(xscale="log")
        ax[0].set(title="Wasserstein Distance - Surface Markers")
        ax[1].set(title="KL Divergence - Surface Markers")
    return corrsDF


# figure 5,7
def EMD_2D(dataset, signal_receptor, target_cells, special_receptor, ax):
    """
    returns list of descending EMD values for specified target cell (2 receptors)
    :param dataset: DataFrame containing the dataset
    :param signal_receptor: Name of the signal receptor
    :param target_cells: Target cell type for analysis
    :param special_receptor: Special receptor to consider
        (optional, used for just calculating distance for 2 receptors)
    :param ax: Matplotlib Axes object for plotting (optional)
    :return:
        List of tuples format: (recep1, recep2, OT value) containing
            optimal transport distances and receptor information
    """
    CITE_DF = importCITE()
    weightDF = convFactCalc(CITE_DF)
    # filter those outliers!
    target_cells_df = dataset[
        (dataset["CellType3"] == target_cells) | (dataset["CellType2"] == target_cells)
    ]
    off_target_cells_df = dataset[
        ~(
            (dataset["CellType3"] == target_cells)
            | (dataset["CellType2"] == target_cells)
        )
    ]
=======
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
        isinstance(i, bool) for i in np.append(targ, offTarg)
    )  # Check that targ and offTarg are only boolean
    assert (
        sum(targ) != 0 and sum(offTarg) != 0
    )  # Check that there are target and off-target cells

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
        if (  # Check these with pre-normalization values
            np.mean(recAbundances[:, rec]) > 5
            and np.mean(recAbundances[targ, rec]) > np.mean(recAbundances[offTarg, rec])
        ):
            targAbun = targNorms[:, rec]
            offTargAbun = offTargNorms[:, rec]

            assert all(
                targAbun == recAbundances[targ, rec] / np.mean(recAbundances[:, rec])
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

    assert all(isinstance(i, bool) for i in np.append(targ, offTarg))
    assert sum(targ) != 0 and sum(offTarg) != 0

    KL_div_vals = np.full((recAbundances.shape[1], recAbundances.shape[1]), np.nan)
    EMD_vals = np.full((recAbundances.shape[1], recAbundances.shape[1]), np.nan)
>>>>>>> main

    targNorms = recAbundances[targ, :] / np.mean(recAbundances, axis=0)
    offTargNorms = recAbundances[offTarg, :] / np.mean(recAbundances, axis=0)

    assert targNorms.shape[0] == sum(targ)
    assert targNorms.shape[0] != recAbundances.shape[0]

    row, col = np.tril_indices(
        recAbundances.shape[1]
    )  # Triangle indices, includes diagonal (k=0 by default)
    for rec1, rec2 in zip(row, col, strict=False):
        if (
            np.mean(recAbundances[:, rec1]) > 5
            and np.mean(recAbundances[:, rec2]) > 5
            and np.mean(recAbundances[targ, rec1])
            > np.mean(recAbundances[offTarg, rec1])
            and np.mean(recAbundances[targ, rec2])
            > np.mean(recAbundances[offTarg, rec2])
        ):
            targAbun1, targAbun2 = targNorms[:, rec1], targNorms[:, rec2]
            offTargAbun1, offTargAbun2 = offTargNorms[:, rec1], offTargNorms[:, rec2]

            assert all(
                targAbun1 == recAbundances[targ, rec1] / np.mean(recAbundances[:, rec1])
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


<<<<<<< HEAD
def calculate_kl_divergence_2D(targCellMark, offTargCellMark):
    """
    calculates the Kullback-Leibler (KL) divergence between two
        probability distributions
    *used in combination with 1D or 2D KL functions
    Armaan: I think this docstring is outdated, as it says that this function is
    used for both 1d and 2d KLs. It actually seems like you can reuse this
    function for 1d and 2d, as KernelDensity.fit() takes in 1d and 2d data.
    :param targCellMark: Target cell marker data
    :param offTargCellMark: Off-target cell marker data
    :return:
    KL_div: KL Divergence value
    """
    kdeTarg = KernelDensity(kernel="gaussian").fit(targCellMark.reshape(-1, 2))
    kdeOffTarg = KernelDensity(kernel="gaussian").fit(offTargCellMark.reshape(-1, 2))
    minVal = np.minimum(targCellMark.min(), offTargCellMark.min()) - 10
    maxVal = np.maximum(targCellMark.max(), offTargCellMark.max()) + 10
    X, Y = np.mgrid[
        minVal : maxVal : ((maxVal - minVal) / 100),
        minVal : maxVal : ((maxVal - minVal) / 100),
    ]
    outcomes = np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1)
    distTarg = np.exp(kdeTarg.score_samples(outcomes))
    distOffTarg = np.exp(kdeOffTarg.score_samples(outcomes))
    KL_div = stats.entropy(
        distOffTarg.flatten() + 1e-200, distTarg.flatten() + 1e-200, base=2
    )
    return KL_div


# figure 5,8
def KL_divergence_2D(dataset, signal_receptor, target_cells, special_receptor, ax):
    """
    Armaan: does this return EMD or just KL? I think either the function name or
    docstring should be changed.
    returns list of descending EMD values for specified target cell (2 receptors)
    :param dataset: DataFrame containing the dataset
    :param signal_receptor: Name of the signal receptor
    :param target_cells: Target cell types for analysis
    :param special_receptor: Special receptor to consider
        (optional, used for just calculating distance for 2 receptors)
    :param ax: Matplotlib Axes object for plotting (optional)
    :return:
    Sorted list of tuples containing KL Divergence values and receptor information
    """
    CITE_DF = importCITE()
    weightDF = convFactCalc(CITE_DF)

    # Armaan: just declare one idx and then use it and its negation to filter
    # the df so you don't repeat code.
    target_cells_df = dataset[
        (dataset["CellType3"] == target_cells) | (dataset["CellType2"] == target_cells)
    ]
    off_target_cells_df = dataset[
        ~(
            (dataset["CellType3"] == target_cells)
            | (dataset["CellType2"] == target_cells)
        )
    ]

    if signal_receptor == "CD122":
        conversion_factor_sig = weightDF.loc[
            weightDF["Receptor"] == "IL2Rb", "Weight"
        ].values[0]
    elif signal_receptor == "CD25":
        conversion_factor_sig = weightDF.loc[
            weightDF["Receptor"] == "IL2Ra", "Weight"
        ].values[0]
    elif signal_receptor == "CD127":
        conversion_factor_sig = weightDF.loc[
            weightDF["Receptor"] == "IL7Ra", "Weight"
        ].values[0]
    else:
        conversion_factor_sig = (
            weightDF.loc[weightDF["Receptor"] == "IL7Ra", "Weight"].values[0]
            + weightDF.loc[weightDF["Receptor"] == "IL2Ra", "Weight"].values[0]
            + weightDF.loc[weightDF["Receptor"] == "IL2Rb", "Weight"].values[0]
        ) / 3

    results = []
    non_signal_receptors = (
        ["CD122", "CD25", "CD127"] if special_receptor is None else [special_receptor]
    )
    for receptor_name in non_signal_receptors:
        target_receptor_counts = target_cells_df[
            [signal_receptor, receptor_name]
        ].values
        off_target_receptor_counts = off_target_cells_df[
            [signal_receptor, receptor_name]
        ].values
        # Armaan: can you declare these mappings once and then use them throughout?
        if receptor_name == "CD122":
            conversion_factor = weightDF.loc[
                weightDF["Receptor"] == "IL2Rb", "Weight"
            ].values[0]
        elif receptor_name == "CD25":
            conversion_factor = weightDF.loc[
                weightDF["Receptor"] == "IL2Ra", "Weight"
            ].values[0]
        elif receptor_name == "CD127":
            conversion_factor = weightDF.loc[
                weightDF["Receptor"] == "IL7Ra", "Weight"
            ].values[0]
        else:
            avg_weight = (
                weightDF.loc[weightDF["Receptor"] == "IL7Ra", "Weight"].values[0]
                + weightDF.loc[weightDF["Receptor"] == "IL2Ra", "Weight"].values[0]
                + weightDF.loc[weightDF["Receptor"] == "IL2Rb", "Weight"].values[0]
            ) / 3
            conversion_factor = avg_weight
        target_receptor_counts[:, 0] *= conversion_factor_sig
        off_target_receptor_counts[:, 0] *= conversion_factor_sig
=======
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
>>>>>>> main

    targNorms = recAbundances[targ, :] / np.mean(recAbundances, axis=0)
    offTargNorms = recAbundances[offTarg, :] / np.mean(recAbundances, axis=0)

    assert targNorms.shape[0] == sum(targ)
    assert targNorms.shape[0] != recAbundances.shape[0]

    for rec1, rec2, rec3 in combinations_with_replacement(
        range(recAbundances.shape[1]), 3
    ):  # 3D triangle (pyramidal?) indices, with replacement includes diagonals
        if (
            np.mean(recAbundances[:, rec1]) > 5
            and np.mean(recAbundances[:, rec2]) > 5
            and np.mean(recAbundances[:, rec3]) > 5
            and np.mean(recAbundances[targ, rec1])
            > np.mean(recAbundances[offTarg, rec1])
            and np.mean(recAbundances[targ, rec2])
            > np.mean(recAbundances[offTarg, rec2])
            and np.mean(recAbundances[targ, rec3])
            > np.mean(recAbundances[offTarg, rec3])
        ):
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

<<<<<<< HEAD
# figure 5
def correlation(cell_type, relevant_epitopes):
    """Calculates the Pearson correlation between two celltypes receptor counts"""
    epitopesList = pd.read_csv("./bicytok/data/epitopeList.csv")
    epitopes = list(epitopesList["Epitope"].unique())
    epitopesDF = calcReceptorAbundances(epitopes, np.array([cell_type]))
    epitopesDF = epitopesDF[epitopesDF["CellType2"] == (cell_type)]
    corr = epitopesDF[relevant_epitopes].corr(method="pearson")
    sorted_corr = corr.stack().sort_values(ascending=False)
    sorted_corr_df = pd.DataFrame({"Correlation": sorted_corr})
    return sorted_corr_df
=======
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
>>>>>>> main
