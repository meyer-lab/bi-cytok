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

        target_receptor_counts[:, 1] *= conversion_factor
        off_target_receptor_counts[:, 1] *= conversion_factor

        average_receptor_counts = np.mean(
            np.concatenate((target_receptor_counts, off_target_receptor_counts)), axis=0
        )
        # Normalize the counts by dividing by the average
        if (
            average_receptor_counts[0] > 5
            and average_receptor_counts[1] > 5
            and np.mean(target_receptor_counts[:, 0])
            > np.mean(off_target_receptor_counts[:, 0])
            and np.mean(target_receptor_counts[:, 1])
            > np.mean(off_target_receptor_counts[:, 1])
        ):
            target_receptor_counts = (
                target_receptor_counts.astype(float) / average_receptor_counts
            )
            off_target_receptor_counts = (
                off_target_receptor_counts.astype(float) / average_receptor_counts
            )
            # Matrix for emd parameter
            M = ot.dist(target_receptor_counts, off_target_receptor_counts)
            # optimal transport distance
            a = (
                np.ones((target_receptor_counts.shape[0],))
                / target_receptor_counts.shape[0]
            )
            b = (
                np.ones((off_target_receptor_counts.shape[0],))
                / off_target_receptor_counts.shape[0]
            )
            optimal_transport = ot.emd2(a, b, M, numItermax=1000000000)
            if special_receptor is not None:
                return optimal_transport  # Return the distance value directly
            results.append((optimal_transport, receptor_name, signal_receptor))
        else:
            results.append((0, receptor_name, signal_receptor))

    # end loop
    sorted_results = sorted(results, reverse=True)
    top_receptor_info = [
        (receptor_name, optimal_transport, signal_receptor)
        for optimal_transport, receptor_name, signal_receptor in sorted_results[:10]
    ]
    # bar graph
    receptor_names = [info[0] for info in top_receptor_info]
    distances = [info[1] for info in top_receptor_info]
    if ax is not None:
        ax.bar(range(len(receptor_names)), distances)
        ax.set_xlabel("Receptor")
        ax.set_ylabel("Distance")
        ax.set_title("Top 5 Receptor Distances (2D)")
        ax.set_xticks(range(len(receptor_names)))
        ax.set_xticklabels(receptor_names, rotation="vertical")

    return sorted_results


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

        target_receptor_counts[:, 1] *= conversion_factor
        off_target_receptor_counts[:, 1] *= conversion_factor

        KL_div = calculate_kl_divergence_2D(
            target_receptor_counts[:, 0:2], off_target_receptor_counts[:, 0:2]
        )
        if np.mean(target_receptor_counts[:, 0]) > np.mean(
            off_target_receptor_counts[:, 0]
        ) and np.mean(target_receptor_counts[:, 1]) > np.mean(
            off_target_receptor_counts[:, 1]
        ):
            if special_receptor is not None:
                return KL_div  # Return the distance value directly
            results.append((KL_div, receptor_name, signal_receptor))
        else:
            results.append((-1, receptor_name, signal_receptor))

    sorted_results = sorted(results, reverse=True)
    top_receptor_info = [
        (receptor_name, KL_div, signal_receptor)
        for KL_div, receptor_name, signal_receptor in sorted_results[:10]
    ]
    # bar graph
    receptor_names = [info[0] for info in top_receptor_info]
    distances = [info[1] for info in top_receptor_info]
    if ax is not None:
        ax.bar(range(len(receptor_names)), distances)
        ax.set_xlabel("Receptor")
        ax.set_ylabel("Distance")
        ax.set_title("Top 5 Receptor Distances (2D)")
        ax.set_xticks(range(len(receptor_names)))
        ax.set_xticklabels(receptor_names, rotation="vertical")

    return sorted_results


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
