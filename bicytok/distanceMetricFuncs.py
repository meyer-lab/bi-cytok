# NOTE: GENERALLY REORGANIZE TO MAKE SENSE
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy import stats
import ot
import ot.plot
from .selectivityFuncs import convFactCalc, getSampleAbundances, optimizeDesign
from os.path import dirname, join
from .imports import importCITE

path_here = dirname(dirname(__file__))

def KL_EMD_1D(ax, targCell, numFactors, RNA=False, offTargState=0) -> pd.DataFrame:
    """
    Finds markers which have average greatest difference (EMD and KL) from other cells
    :param ax: Axes to plot on
    :param targCell: Target cell type for analysis
    :param numFactors: Number of top factors to consider
    :param RNA: Boolean flag indicating RNA data (optional)
    :param offTargState: State of off-target comparison (0 for all non-memory Tregs, 1 for all non-Tregs, 2 for naive Tregs)
    :return:
        corrsDF: DataFrame containing marker information and their Wasserstein Distance and KL Divergence values
    """
    CITE_DF = importCITE()
    markerDF = pd.DataFrame(columns=["Marker", "Cell Type", "Amount"])
    for marker in CITE_DF.loc[:, ((CITE_DF.columns != 'CellType1') & (CITE_DF.columns != 'CellType2') & (CITE_DF.columns != 'CellType3') & (CITE_DF.columns != 'Cell'))].columns:
        markAvg = np.mean(CITE_DF[marker].values)
        if markAvg > 0.0001:
            targCellMark = CITE_DF.loc[CITE_DF["CellType3"] == targCell][marker].values / markAvg
            # Compare to all non-memory Tregs
            if offTargState == 0:
                offTargCellMark = CITE_DF.loc[CITE_DF["CellType3"] != targCell][marker].values / markAvg
            # Compare to all non-Tregs
            elif offTargState == 1:
                offTargCellMark = CITE_DF.loc[CITE_DF["CellType2"] != "Treg"][marker].values / markAvg
            # Compare to naive Tregs
            elif offTargState == 2:
                offTargCellMark = CITE_DF.loc[CITE_DF["CellType3"] == "Treg Naive"][marker].values / markAvg
            if np.mean(targCellMark) > np.mean(offTargCellMark):
                kdeTarg = KernelDensity(kernel='gaussian').fit(targCellMark.reshape(-1, 1))
                kdeOffTarg = KernelDensity(kernel='gaussian').fit(offTargCellMark.reshape(-1, 1))
                minVal = np.minimum(targCellMark.min(), offTargCellMark.min()) - 10
                maxVal = np.maximum(targCellMark.max(), offTargCellMark.max()) + 10
                outcomes = np.arange(minVal, maxVal + 1).reshape(-1, 1)
                distTarg = np.exp(kdeTarg.score_samples(outcomes))
                distOffTarg = np.exp(kdeOffTarg.score_samples(outcomes))
                KL_div = stats.entropy(distOffTarg.flatten() + 1e-200, distTarg.flatten() + 1e-200, base=2)
                markerDF = pd.concat([markerDF, pd.DataFrame({"Marker": [marker], "Wasserstein Distance": stats.wasserstein_distance(targCellMark, offTargCellMark), "KL Divergence": KL_div})])

    corrsDF = pd.DataFrame()
    for i, distance in enumerate(["Wasserstein Distance", "KL Divergence"]):
        ratioDF = markerDF.sort_values(by=distance)
        posCorrs = ratioDF.tail(numFactors).Marker.values
        corrsDF = pd.concat([corrsDF, pd.DataFrame({"Distance": distance, "Marker": posCorrs})])
        markerDF = markerDF.loc[markerDF["Marker"].isin(posCorrs)]
        sns.barplot(data=ratioDF.tail(numFactors), y="Marker", x=distance, ax=ax[i], color='k')
        ax[i].set(xscale="log")
        ax[0].set(title="Wasserstein Distance - Surface Markers")
        ax[1].set(title="KL Divergence - Surface Markers")
    return corrsDF

def get_conversion_factor(weightDF, receptor_name):
    '''
    conversion factors used for citeseq dataset
    :param weightDF: DataFrame containing weight information for receptors
    :param receptor_name: Name of the receptor for which the conversion factor is needed
    :return:
        conversion_factor: Conversion factor for the specified receptor
    '''
    IL2Rb_factor = weightDF.loc[weightDF['Receptor'] == 'IL2Rb', 'Weight'].values[0]
    IL7Ra_factor = weightDF.loc[weightDF['Receptor'] == 'IL7Ra', 'Weight'].values[0]
    IL2Ra_factor = weightDF.loc[weightDF['Receptor'] == 'IL2Ra', 'Weight'].values[0]
    if receptor_name == 'CD122':
        return IL2Rb_factor
    elif receptor_name == 'CD25':
        return IL2Ra_factor
    elif receptor_name == 'CD127':
        return IL7Ra_factor
    else: return (IL7Ra_factor + IL2Ra_factor + IL2Rb_factor) / 3

def EMD_2D(dataset, signal_receptor, target_cells, special_receptor, ax):
    '''
    returns list of descending EMD values for specified target cell (2 receptors)
    :param dataset: DataFrame containing the dataset
    :param signal_receptor: Name of the signal receptor
    :param target_cells: Target cell type for analysis
    :param special_receptor: Special receptor to consider (optional, used for just calculating distance for 2 receptors)
    :param ax: Matplotlib Axes object for plotting (optional)
    :return:
        List of tuples format: (recep1, recep2, OT value) containing optimal transport distances and receptor information
    '''
    CITE_DF = importCITE()
    weightDF = convFactCalc(CITE_DF)
    # filter those outliers! 
    target_cells_df = dataset[(dataset['CellType3'] == target_cells) | (dataset['CellType2'] == target_cells)]
    off_target_cells_df = dataset[~((dataset['CellType3'] == target_cells) | (dataset['CellType2'] == target_cells))]

    if signal_receptor == 'CD122':
        conversion_factor_sig = weightDF.loc[weightDF['Receptor'] == 'IL2Rb', 'Weight'].values[0]
    elif signal_receptor == 'CD25':
        conversion_factor_sig = weightDF.loc[weightDF['Receptor'] == 'IL2Ra', 'Weight'].values[0]
    elif signal_receptor == 'CD127':
        conversion_factor_sig = weightDF.loc[weightDF['Receptor'] == 'IL7Ra', 'Weight'].values[0]
    else:
        conversion_factor_sig = (weightDF.loc[weightDF['Receptor'] == 'IL7Ra', 'Weight'].values[0] + 
                                 weightDF.loc[weightDF['Receptor'] == 'IL2Ra', 'Weight'].values[0] + 
                                 weightDF.loc[weightDF['Receptor'] == 'IL2Rb', 'Weight'].values[0]) / 3

    results = []
    non_signal_receptors = ['CD122', 'CD25', 'CD127'] if special_receptor is None else [special_receptor]
    for receptor_name in non_signal_receptors:
        target_receptor_counts = target_cells_df[[signal_receptor, receptor_name]].values
        off_target_receptor_counts = off_target_cells_df[[signal_receptor, receptor_name]].values
        if receptor_name == 'CD122':
            conversion_factor = weightDF.loc[weightDF['Receptor'] == 'IL2Rb', 'Weight'].values[0]
        elif receptor_name == 'CD25':
            conversion_factor = weightDF.loc[weightDF['Receptor'] == 'IL2Ra', 'Weight'].values[0]
        elif receptor_name == 'CD127':
            conversion_factor = weightDF.loc[weightDF['Receptor'] == 'IL7Ra', 'Weight'].values[0]
        else:
            avg_weight = (weightDF.loc[weightDF['Receptor'] == 'IL7Ra', 'Weight'].values[0] + 
                        weightDF.loc[weightDF['Receptor'] == 'IL2Ra', 'Weight'].values[0] + 
                        weightDF.loc[weightDF['Receptor'] == 'IL2Rb', 'Weight'].values[0]) / 3
            conversion_factor = avg_weight
        target_receptor_counts[:, 0] *= conversion_factor_sig
        off_target_receptor_counts[:, 0] *= conversion_factor_sig

        target_receptor_counts[:, 1] *= conversion_factor
        off_target_receptor_counts[:, 1] *= conversion_factor

        average_receptor_counts = np.mean(np.concatenate((target_receptor_counts, off_target_receptor_counts)), axis=0)
        # Normalize the counts by dividing by the average
        if average_receptor_counts[0] > 5 and average_receptor_counts[1] > 5 and np.mean(target_receptor_counts[:, 0]) > np.mean(off_target_receptor_counts[:, 0]) and np.mean(target_receptor_counts[:, 1]) > np.mean(off_target_receptor_counts[:, 1]):
            target_receptor_counts = target_receptor_counts.astype(float) / average_receptor_counts
            off_target_receptor_counts = off_target_receptor_counts.astype(float) / average_receptor_counts
            # Matrix for emd parameter
            M = ot.dist(target_receptor_counts, off_target_receptor_counts)
            # optimal transport distance
            a = np.ones((target_receptor_counts.shape[0],)) / target_receptor_counts.shape[0]
            b = np.ones((off_target_receptor_counts.shape[0],)) / off_target_receptor_counts.shape[0]
            optimal_transport = ot.emd2(a, b, M, numItermax=1000000000)
            if special_receptor is not None:
                return optimal_transport  # Return the distance value directly
            results.append((optimal_transport, receptor_name, signal_receptor))
        else:
            results.append((0, receptor_name, signal_receptor))

    # end loop
    sorted_results = sorted(results, reverse=True)
    top_receptor_info = [(receptor_name, optimal_transport, signal_receptor) for optimal_transport, receptor_name, signal_receptor in sorted_results[:10]]
    # bar graph 
    receptor_names = [info[0] for info in top_receptor_info]
    distances = [info[1] for info in top_receptor_info]
    if ax is not None:
        ax.bar(range(len(receptor_names)), distances)
        ax.set_xlabel('Receptor')
        ax.set_ylabel('Distance')
        ax.set_title('Top 5 Receptor Distances (2D)')
        ax.set_xticks(range(len(receptor_names)))
        ax.set_xticklabels(receptor_names, rotation='vertical')
    
    return sorted_results

def EMD_3D(dataset1, target_cells, ax=None):
    '''
    returns list of descending EMD values for specified target cell (3 receptors)
    returns list of descending EMD values for specified target cell (3 receptors)
    :param dataset1: DataFrame containing the dataset
    :param target_cells: Target cell type for analysis
    :param ax: Matplotlib Axes object for plotting (optional)
    :return:
    List of tuples (format: (recep1, recep2, recep 3, OT value) containing optimal transport distances and receptor information for 3D analysis
    
    '''
    CITE_DF = importCITE()

    weightDF = convFactCalc(CITE_DF)
    exclude_columns = ['CellType1', 'CellType2', 'CellType3', 'Cell']
    threshold_multiplier = 5
    # Calculate the mean and standard deviation for each numeric column
    numeric_columns = [col for col in dataset1.columns if col not in exclude_columns]
    column_means = dataset1[numeric_columns].mean()
    column_stddevs = dataset1[numeric_columns].std()
    # Identify outliers for each numeric column
    outliers = {}
    for column in numeric_columns:
        threshold = column_means[column] + threshold_multiplier * column_stddevs[column]
        outliers[column] = dataset1[column] > threshold
    # Create a mask to filter rows with outliers
    outlier_mask = pd.DataFrame(outliers)
    dataset = dataset1[~outlier_mask.any(axis=1)]
    # receptor_names = [col for col in dataset.columns if col not in exclude_columns]
    results = []
    target_cells_df = dataset[(dataset['CellType3'] == target_cells) | (dataset['CellType2'] == target_cells) | (dataset['CellType1'] == target_cells)]
    off_target_cells_df = dataset[~((dataset['CellType3'] == target_cells) | (dataset['CellType2'] == target_cells) | (dataset['CellType1'] == target_cells))]
    receptor_names = [col for col in dataset.columns if col not in exclude_columns and
                      np.mean(target_cells_df[col]) > np.mean(off_target_cells_df[col]) and
                   np.mean(target_cells_df[col]) > 5]
    
    for receptor1_name in receptor_names:
        for receptor2_name in receptor_names:
            for receptor3_name in receptor_names:
                if receptor1_name != receptor2_name and receptor1_name != receptor3_name and receptor2_name != receptor3_name:
                    # Get on and off-target counts for receptor1 2 and 3
                    receptor1_on_target_counts = target_cells_df[receptor1_name].values
                    receptor1_off_target_counts = off_target_cells_df[receptor1_name].values
                    receptor2_on_target_counts = target_cells_df[receptor2_name].values
                    receptor2_off_target_counts = off_target_cells_df[receptor2_name].values
                    receptor3_on_target_counts = target_cells_df[receptor3_name].values
                    receptor3_off_target_counts = off_target_cells_df[receptor3_name].values
                    conversion_factor_receptor1 = get_conversion_factor(weightDF, receptor1_name)
                    conversion_factor_receptor2 = get_conversion_factor(weightDF, receptor2_name)
                    conversion_factor_receptor3 = get_conversion_factor(weightDF, receptor3_name)
                    # Apply the conversion factors to the counts
                    receptor1_on_target_counts = receptor1_on_target_counts * conversion_factor_receptor1
                    receptor1_off_target_counts = receptor1_off_target_counts * conversion_factor_receptor1
                    receptor2_on_target_counts = receptor2_on_target_counts * conversion_factor_receptor2
                    receptor2_off_target_counts = receptor2_off_target_counts * conversion_factor_receptor2
                    receptor3_on_target_counts = receptor3_on_target_counts * conversion_factor_receptor3
                    receptor3_off_target_counts = receptor3_off_target_counts * conversion_factor_receptor3
                    average_receptor_counts_1_on = np.mean(receptor1_on_target_counts)
                    average_receptor_counts_1_off = np.mean(receptor1_off_target_counts)
                    average_receptor_counts_2_on = np.mean(receptor2_on_target_counts)
                    average_receptor_counts_2_off = np.mean(receptor2_off_target_counts)
                    average_receptor_counts_3_on = np.mean(receptor3_on_target_counts)
                    average_receptor_counts_3_off = np.mean(receptor3_off_target_counts)
                    average_receptor_counts_1 = np.mean(np.concatenate((receptor1_on_target_counts, receptor1_off_target_counts)), axis=0)
                    average_receptor_counts_2 = np.mean(np.concatenate((receptor2_on_target_counts, receptor2_off_target_counts)), axis=0)
                    average_receptor_counts_3 = np.mean(np.concatenate((receptor3_on_target_counts, receptor3_off_target_counts)), axis=0)
                    if average_receptor_counts_1_on > 5 and average_receptor_counts_2_on > 5 and average_receptor_counts_3_on > 5 and average_receptor_counts_1_on > average_receptor_counts_1_off and average_receptor_counts_2_on > average_receptor_counts_2_off and average_receptor_counts_3_on > average_receptor_counts_3_off:
                        receptor1_on_target_counts = receptor1_on_target_counts.astype(float) / average_receptor_counts_1
                        receptor1_off_target_counts = receptor1_off_target_counts.astype(float) / average_receptor_counts_1
                        receptor2_on_target_counts = receptor2_on_target_counts.astype(float) / average_receptor_counts_2
                        receptor2_off_target_counts = receptor2_off_target_counts.astype(float) / average_receptor_counts_2
                        receptor3_on_target_counts = receptor3_on_target_counts.astype(float) / average_receptor_counts_3
                        receptor3_off_target_counts = receptor3_off_target_counts.astype(float) / average_receptor_counts_3
                        # Calculate the EMD between on-target and off-target counts for both receptors # change this so its two [||]
                        on_target_counts = np.concatenate((receptor1_on_target_counts[:, np.newaxis], receptor2_on_target_counts[:, np.newaxis], receptor3_on_target_counts[:, np.newaxis]), axis=1)
                        off_target_counts = np.concatenate((receptor1_off_target_counts[:, np.newaxis], receptor2_off_target_counts[:, np.newaxis], receptor3_off_target_counts[:, np.newaxis]), axis=1)
                        average_receptor_counts = np.mean(np.concatenate((on_target_counts, off_target_counts)), axis=0)
                        on_target_counts = on_target_counts.astype(float) / average_receptor_counts
                        off_target_counts = off_target_counts.astype(float) / average_receptor_counts
                        M = ot.dist(on_target_counts, off_target_counts)
                        a = np.ones(on_target_counts.shape[0]) / on_target_counts.shape[0]
                        b = np.ones(off_target_counts.shape[0]) / off_target_counts.shape[0]
                        optimal_transport = ot.emd2(a, b, M, numItermax=10000000)
                        results.append((optimal_transport, receptor1_name, receptor2_name, receptor3_name))
                        print ('ot:', optimal_transport)
                    else:
                        results.append((0, receptor1_name, receptor2_name, receptor3_name))

    sorted_results = sorted(results, reverse=True)
    top_receptor_info = [(receptor1_name, receptor2_name, receptor3_name, optimal_transport) for optimal_transport, receptor1_name, receptor2_name, receptor3_name in sorted_results[:10]]
    print ('top 10 dist:', top_receptor_info)
    receptor_pairs = [(info[0], info[1], info[2]) for info in top_receptor_info]
    distances = [info[3] for info in top_receptor_info]
    if ax is not None:
        ax.bar(range(len(receptor_pairs)), distances)
        ax.set_xlabel('Receptor Pair', fontsize=14)
        ax.set_ylabel('Distance', fontsize=14)
        ax.set_title(f'Top 10 Receptor Pair Distances (3D) for {target_cells}', fontsize=14)
        ax.set_xticks(range(len(receptor_pairs)))
        ax.set_xticklabels([f"{pair[0]} - {pair[1]} - {pair[2]}" for pair in receptor_pairs], rotation='vertical', fontsize=14) 
    return sorted_results

def calculate_kl_divergence_2D(targCellMark, offTargCellMark):
    '''  
    calculates the Kullback-Leibler (KL) divergence between two probability distributions
    *used in combination with 1D or 2D KL functions
    :param targCellMark: Target cell marker data
    :param offTargCellMark: Off-target cell marker data
    :return:
    KL_div: KL Divergence value
    '''
    kdeTarg = KernelDensity(kernel='gaussian').fit(targCellMark.reshape(-1, 2))
    kdeOffTarg = KernelDensity(kernel='gaussian').fit(offTargCellMark.reshape(-1, 2))
    minVal = np.minimum(targCellMark.min(), offTargCellMark.min()) - 10
    maxVal = np.maximum(targCellMark.max(), offTargCellMark.max()) + 10
    X, Y = np.mgrid[minVal:maxVal:((maxVal-minVal) / 100), minVal:maxVal:((maxVal-minVal) / 100)]
    outcomes = np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1)
    distTarg = np.exp(kdeTarg.score_samples(outcomes))
    distOffTarg = np.exp(kdeOffTarg.score_samples(outcomes))
    KL_div = stats.entropy(distOffTarg.flatten() + 1e-200, distTarg.flatten() + 1e-200, base=2)
    return KL_div

def KL_divergence_2D(dataset, signal_receptor, target_cells, special_receptor, ax):
        
    '''
    returns list of descending EMD values for specified target cell (2 receptors) 
    :param dataset: DataFrame containing the dataset
    :param signal_receptor: Name of the signal receptor
    :param target_cells: Target cell type for analysis
    :param special_receptor: Special receptor to consider (optional, used for just calculating distance for 2 receptors)
    :param ax: Matplotlib Axes object for plotting (optional)
    :return:
    Sorted list of tuples containing KL Divergence values and receptor information
    '''
    CITE_DF = importCITE()
    weightDF = convFactCalc(CITE_DF)

    target_cells_df = dataset[(dataset['CellType3'] == target_cells) | (dataset['CellType2'] == target_cells)]
    off_target_cells_df = dataset[~((dataset['CellType3'] == target_cells) | (dataset['CellType2'] == target_cells))]

    if signal_receptor == 'CD122':
        conversion_factor_sig = weightDF.loc[weightDF['Receptor'] == 'IL2Rb', 'Weight'].values[0]
    elif signal_receptor == 'CD25':
        conversion_factor_sig = weightDF.loc[weightDF['Receptor'] == 'IL2Ra', 'Weight'].values[0]
    elif signal_receptor == 'CD127':
        conversion_factor_sig = weightDF.loc[weightDF['Receptor'] == 'IL7Ra', 'Weight'].values[0]
    else:
        conversion_factor_sig = (weightDF.loc[weightDF['Receptor'] == 'IL7Ra', 'Weight'].values[0] + 
                                 weightDF.loc[weightDF['Receptor'] == 'IL2Ra', 'Weight'].values[0] + 
                                 weightDF.loc[weightDF['Receptor'] == 'IL2Rb', 'Weight'].values[0]) / 3

    results = []
    non_signal_receptors = ['CD122', 'CD25', 'CD127'] if special_receptor is None else [special_receptor]
    for receptor_name in non_signal_receptors:
        target_receptor_counts = target_cells_df[[signal_receptor, receptor_name]].values
        off_target_receptor_counts = off_target_cells_df[[signal_receptor, receptor_name]].values
        if receptor_name == 'CD122':
            conversion_factor = weightDF.loc[weightDF['Receptor'] == 'IL2Rb', 'Weight'].values[0]
        elif receptor_name == 'CD25':
            conversion_factor = weightDF.loc[weightDF['Receptor'] == 'IL2Ra', 'Weight'].values[0]
        elif receptor_name == 'CD127':
            conversion_factor = weightDF.loc[weightDF['Receptor'] == 'IL7Ra', 'Weight'].values[0]
        else:
            avg_weight = (weightDF.loc[weightDF['Receptor'] == 'IL7Ra', 'Weight'].values[0] + 
                        weightDF.loc[weightDF['Receptor'] == 'IL2Ra', 'Weight'].values[0] + 
                        weightDF.loc[weightDF['Receptor'] == 'IL2Rb', 'Weight'].values[0]) / 3
            conversion_factor = avg_weight
        target_receptor_counts[:, 0] *= conversion_factor_sig
        off_target_receptor_counts[:, 0] *= conversion_factor_sig

        target_receptor_counts[:, 1] *= conversion_factor
        off_target_receptor_counts[:, 1] *= conversion_factor

        KL_div = calculate_kl_divergence_2D(target_receptor_counts[:, 0:2], off_target_receptor_counts[:, 0:2])
        if np.mean(target_receptor_counts[:, 0]) > np.mean(off_target_receptor_counts[:, 0]) and np.mean(target_receptor_counts[:, 1]) > np.mean(off_target_receptor_counts[:, 1]):
            if special_receptor is not None:
                return KL_div  # Return the distance value directly
            results.append((KL_div, receptor_name, signal_receptor))
        else:
            results.append((-1, receptor_name, signal_receptor))

    sorted_results = sorted(results, reverse=True)
    top_receptor_info = [(receptor_name, KL_div, signal_receptor) for KL_div, receptor_name, signal_receptor in sorted_results[:10]]
    # bar graph 
    receptor_names = [info[0] for info in top_receptor_info]
    distances = [info[1] for info in top_receptor_info]
    if ax is not None:
        ax.bar(range(len(receptor_names)), distances)
        ax.set_xlabel('Receptor')
        ax.set_ylabel('Distance')
        ax.set_title('Top 5 Receptor Distances (2D)')
        ax.set_xticks(range(len(receptor_names)))
        ax.set_xticklabels(receptor_names, rotation='vertical')

    return sorted_results

def correlation(cell_type, relevant_epitopes):
    '''calculates the Pearson correlation between two celltypes receptor counts'''
    epitopesList = pd.read_csv("./bicytok/data/epitopeList.csv")
    epitopes = list(epitopesList['Epitope'].unique())
    epitopesDF = getSampleAbundances(epitopes, np.array([cell_type]))
    epitopesDF = epitopesDF[epitopesDF['CellType2'] == (cell_type)]
    corr = epitopesDF[relevant_epitopes].corr(method='pearson')
    sorted_corr = corr.stack().sort_values(ascending=False)
    sorted_corr_df = pd.DataFrame({'Correlation': sorted_corr})
    return sorted_corr_df