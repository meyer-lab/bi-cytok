# NOTE: GENERALLY REORGANIZE TO MAKE SENSE

# NOTE: SEPARATE INTO FIGURE GENERATION IN FIGURE AND CALCULATIONS HERE @MYSELF
def KL_EMD_1D(ax, targCell, numFactors, RNA=False, offTargState=0) -> pd.DataFrame:
    """Finds markers which have average greatest difference from other cells"""
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

# NOTE: RENAME FUNCTION?
def common_code(weightDF, dataset, signal_receptor, target_cells):
    '''used in EMD2D and KL2D to make target and off target cell dfs'''
    non_signal_receptors = []
    for column in dataset.columns:
        if column != signal_receptor and column not in ['CellType1', 'CellType2', 'CellType3']:
            non_signal_receptors.append(column)
    target_cells_df = dataset[(dataset['CellType3'] == target_cells) | (dataset['CellType2'] == target_cells) | (dataset['CellType1'] == target_cells)]
    off_target_cells_df = dataset[~((dataset['CellType3'] == target_cells) | (dataset['CellType2'] == target_cells) | (dataset['CellType1'] == target_cells))]
    conversion_factor_sig = get_conversion_factor(weightDF, signal_receptor)
    return target_cells_df, off_target_cells_df, non_signal_receptors, conversion_factor_sig

# NOTE: MOVE THIS WITH EMD KL STUFF
def get_conversion_factor(weightDF, receptor_name):
    '''conversion factors used for citeseq dataset'''
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

def EMD_2D(dataset, signal_receptor, target_cells, ax):
    '''returns list of descending EMD values for specified target cell (2 receptors) '''
    weightDF = convFactCalc()
    # filter those outliers! 
    exclude_columns = ['CellType1', 'CellType2', 'CellType3', 'Cell']
    # Define a threshold multiplier to identify outliers (e.g., 3 times the standard deviation)
    threshold_multiplier = 5
    # Calculate the mean and standard deviation for each numeric column
    numeric_columns = [col for col in dataset.columns if col not in exclude_columns]
    column_means = dataset[numeric_columns].mean()
    column_stddevs = dataset[numeric_columns].std()
    # Identify outliers for each numeric column
    outliers = {}
    for column in numeric_columns:
        threshold = column_means[column] + threshold_multiplier * column_stddevs[column]
        outliers[column] = dataset[column] > threshold
    outlier_mask = pd.DataFrame(outliers)
    filtered_dataset = dataset[~outlier_mask.any(axis=1)]
    target_cells_df, off_target_cells_df, non_signal_receptors, conversion_factor_sig = common_code(weightDF, filtered_dataset, signal_receptor, target_cells)
    results = []
    for receptor_name in non_signal_receptors:
        target_receptor_counts = target_cells_df[[signal_receptor, receptor_name]].values
        off_target_receptor_counts = off_target_cells_df[[signal_receptor, receptor_name]].values
        conversion_factor = get_conversion_factor(weightDF, receptor_name)
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
    '''returns list of descending EMD values for specified target cell (3 receptors)'''

    weightDF = convFactCalc()
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

# NOTE: IS THIS NOT JUST A USE-CASE OF EMD_2D?
def EMD_2D_pair(dataset, target_cells, signal_receptor, special_receptor):
    weightDF = convFactCalc()
    # target and off-target cells
    IL2Rb_factor = weightDF.loc[weightDF['Receptor'] == 'IL2Rb', 'Weight'].values[0]
    IL7Ra_factor = weightDF.loc[weightDF['Receptor'] == 'IL7Ra', 'Weight'].values[0]
    IL2Ra_factor = weightDF.loc[weightDF['Receptor'] == 'IL2Ra', 'Weight'].values[0]

    target_cells_df = dataset[(dataset['CellType3'] == target_cells) | (dataset['CellType2'] == target_cells)]
    off_target_cells_df = dataset[~((dataset['CellType3'] == target_cells) | (dataset['CellType2'] == target_cells))]
    
    if signal_receptor == 'CD122':
        conversion_factor_sig = IL2Rb_factor
    elif signal_receptor == 'CD25':
        conversion_factor_sig = IL2Ra_factor
    elif signal_receptor == 'CD127':
        conversion_factor_sig = IL7Ra_factor
    else:
        conversion_factor_sig = (IL7Ra_factor+IL2Ra_factor+IL2Rb_factor)/3
    
    
    target_receptor_counts = target_cells_df[[signal_receptor, special_receptor]].values
    off_target_receptor_counts = off_target_cells_df[[signal_receptor, special_receptor]].values

    if special_receptor == 'CD122':
        conversion_factor = IL2Rb_factor
    elif special_receptor == 'CD25':
        conversion_factor = IL2Ra_factor
    elif special_receptor == 'CD127':
        conversion_factor = IL7Ra_factor
    else:
        conversion_factor = (IL7Ra_factor+IL2Ra_factor+IL2Rb_factor)/3

    target_receptor_counts[:, 0] *= conversion_factor_sig
    off_target_receptor_counts[:, 0] *= conversion_factor_sig

    target_receptor_counts[:, 1] *= conversion_factor
    off_target_receptor_counts[:, 1] *= conversion_factor
        
    average_receptor_counts = np.mean(np.concatenate((target_receptor_counts, off_target_receptor_counts)), axis=0)
    print(np.concatenate((target_receptor_counts, off_target_receptor_counts)))

    # Normalize the counts by dividing by the average
    target_receptor_counts = target_receptor_counts.astype(float) / average_receptor_counts
    off_target_receptor_counts = off_target_receptor_counts.astype(float) / average_receptor_counts
        
    # Matrix for emd parameter
    M = ot.dist(target_receptor_counts, off_target_receptor_counts)
    # optimal transport distance
    a = np.ones((target_receptor_counts.shape[0],)) / target_receptor_counts.shape[0]
    b = np.ones((off_target_receptor_counts.shape[0],)) / off_target_receptor_counts.shape[0]
    optimal_transport = ot.emd2(a, b, M, numItermax=10000000)

    print('OT', optimal_transport)
    return optimal_transport

# NOTE: IS THIS NOT JUST A USE-CASE OF KL_divergence_2D?
def KL_divergence_2D_pair(dataset, target_cells, signal_receptor, special_receptor):
    weightDF = convFactCalc()
    # target and off-target cells
    IL2Rb_factor = weightDF.loc[weightDF['Receptor'] == 'IL2Rb', 'Weight'].values[0]
    IL7Ra_factor = weightDF.loc[weightDF['Receptor'] == 'IL7Ra', 'Weight'].values[0]
    IL2Ra_factor = weightDF.loc[weightDF['Receptor'] == 'IL2Ra', 'Weight'].values[0]

    target_cells_df = dataset[(dataset['CellType3'] == target_cells) | (dataset['CellType2'] == target_cells)]
    off_target_cells_df = dataset[~((dataset['CellType3'] == target_cells) | (dataset['CellType2'] == target_cells))]
    
    if signal_receptor == 'CD122':
        conversion_factor_sig = IL2Rb_factor
    elif signal_receptor == 'CD25':
        conversion_factor_sig = IL2Ra_factor
    elif signal_receptor == 'CD127':
        conversion_factor_sig = IL7Ra_factor
    else:
        conversion_factor_sig = (IL7Ra_factor+IL2Ra_factor+IL2Rb_factor)/3
    
    
    target_receptor_counts = target_cells_df[[signal_receptor, special_receptor]].values
    off_target_receptor_counts = off_target_cells_df[[signal_receptor, special_receptor]].values

    if special_receptor == 'CD122':
        conversion_factor = IL2Rb_factor
    elif special_receptor == 'CD25':
        conversion_factor = IL2Ra_factor
    elif special_receptor == 'CD127':
        conversion_factor = IL7Ra_factor
    else:
        conversion_factor = (IL7Ra_factor+IL2Ra_factor+IL2Rb_factor)/3

    target_receptor_counts[:, 0] *= conversion_factor_sig
    off_target_receptor_counts[:, 0] *= conversion_factor_sig

    target_receptor_counts[:, 1] *= conversion_factor
    off_target_receptor_counts[:, 1] *= conversion_factor
        
    average_receptor_counts = np.mean(np.concatenate((target_receptor_counts, off_target_receptor_counts)), axis=0)

    # Normalize the counts by dividing by the average
    target_receptor_counts = target_receptor_counts.astype(float) / average_receptor_counts
    off_target_receptor_counts = off_target_receptor_counts.astype(float) / average_receptor_counts
        
    KL_div = calculate_kl_divergence_2D(target_receptor_counts[:, 1], off_target_receptor_counts[:, 1])
    
    print('KL', KL_div)
    return KL_div

# NOTE: SHOULD IT BE A FUNCTION OR A FIGURE?
def EMD_KL_clustermap(dataset):
    '''turns datasets from EMD and KL into clustermap'''
    dataset = dataset.fillna(0)
    return (sns.clustermap(dataset, cmap='bwr', figsize=(10,10), annot_kws={'fontsize': 16}))

def calculate_kl_divergence_2D(targCellMark, offTargCellMark):
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

# NOTE: MAYBE REMOVE IF ABOVE DOES THE SAME THING AS ABOVE? SHOULD IT BE A FUNCTION OR A FIGURE?
def EMD_1D(dataset, target_cells, ax):
    weightDF = convFactCalc()
    # target and off-target cells
    IL2Rb_factor = weightDF.loc[weightDF['Receptor'] == 'IL2Rb', 'Weight'].values[0]
    IL7Ra_factor = weightDF.loc[weightDF['Receptor'] == 'IL7Ra', 'Weight'].values[0]
    IL2Ra_factor = weightDF.loc[weightDF['Receptor'] == 'IL2Ra', 'Weight'].values[0]
    
    receptorsdf = []
    for column in dataset.columns:
        if column not in ['CellType1', 'CellType2', 'CellType3']:
            receptorsdf.append(column)

    results = []
    target_cells_df = dataset[(dataset['CellType3'] == target_cells) | (dataset['CellType2'] == target_cells)]
    off_target_cells_df = dataset[~((dataset['CellType3'] == target_cells) | (dataset['CellType2'] == target_cells))]

    for receptor_name in receptorsdf:
        target_receptor_counts = target_cells_df[[receptor_name]].values
        off_target_receptor_counts = off_target_cells_df[[receptor_name]].values
        if receptor_name == 'CD122':
            conversion_factor = IL2Rb_factor
        elif receptor_name == 'CD25':
            conversion_factor = IL2Ra_factor
        elif receptor_name == 'CD127':
            conversion_factor = IL7Ra_factor
        else:
            conversion_factor = (IL7Ra_factor+IL2Ra_factor+IL2Rb_factor)/3

        target_receptor_counts = target_receptor_counts.astype(float) * conversion_factor
        off_target_receptor_counts = off_target_receptor_counts.astype(float) * conversion_factor
        
       
        average_receptor_counts = np.mean(np.concatenate((target_receptor_counts, off_target_receptor_counts)), axis=0)

        #Normalize the counts by dividing by the average
        target_receptor_counts = target_receptor_counts.astype(float) / average_receptor_counts
        off_target_receptor_counts = off_target_receptor_counts.astype(float) / average_receptor_counts

        # Matrix for emd parameter
        M = ot.dist(target_receptor_counts, off_target_receptor_counts)
        # optimal transport distance
        a = np.ones((target_receptor_counts.shape[0],)) / target_receptor_counts.shape[0]
        b = np.ones((off_target_receptor_counts.shape[0],)) / off_target_receptor_counts.shape[0] 
        optimal_transport = ot.emd2(a, b, M, numItermax=10000000)
        if np.mean(target_receptor_counts) > np.mean(off_target_receptor_counts):
            results.append((optimal_transport, receptor_name))
    # end loop
    sorted_results = sorted(results, reverse=True)
    top_receptor_info = [(receptor_name, optimal_transport) for optimal_transport, receptor_name in sorted_results[:5]]    
    # bar graph 
    
    receptor_names = [info[0] for info in top_receptor_info]
    distances = [info[1] for info in top_receptor_info]

    ax.bar(range(len(receptor_names)), distances)
    ax.set_xlabel('Receptor')
    ax.set_ylabel('Distance')
    ax.set_title('Top 5 Receptor Distances (1D)')
    ax.set_xticks(range(len(receptor_names)))
    ax.set_xticklabels(receptor_names, rotation='vertical')
    print('The 5 receptors which achieve the greatest positive distance from target-off-target cells are:', top_receptor_info)
    return sorted_results

# NOTE: DO YOU NEED THIS? SHOULD IT BE A FUNCTION OR A FIGURE?
def EMD1Dvs2D_Analysis(receptor_names, target_cells, signal_receptor, dataset, ax1, ax2, ax3, ax4):
    filtered_data_1D = []
    filtered_data_2D = []
    filtered_data_selectivity = []
    
    EMD1D = EMD_1D(dataset, target_cells, ax1)
    for value, receptor in EMD1D:
        if receptor in receptor_names:
            filtered_data_1D.append((receptor, value))
    EMD2D = EMD_2D(dataset, signal_receptor, target_cells, ax2)
    for value, receptor in EMD2D:
        if receptor in receptor_names:
            filtered_data_2D.append((receptor, value))
    

    cell_types = set(dataset['CellType1']).union(dataset['CellType2']).union(dataset['CellType3'])
    offtarg_cell_types = [cell_type for cell_type in cell_types if cell_type != target_cells]
    epitopes = [column for column in dataset.columns if column not in ['CellType1', 'CellType2', 'CellType3']]
    # below must be changed depending on target cell, celltype3 for treg mem, celltype 2 for treg 
    epitopesDF = getSampleAbundances(epitopes, cell_types, "CellType2") 
    dose = 1
    valency = 2
    
    for receptor_name in receptor_names:
        print("Receptor name:", receptor_name)
        optParams1 = optimizeDesign(signal_receptor, receptor_name, target_cells, offtarg_cell_types, epitopesDF, dose, valency)
        selectivity = 1/optParams1[0]
        filtered_data_selectivity.append([receptor_name, selectivity])
    

    # should ensure order is the same 
    filtered_data_1D.sort(key=lambda x: x[0])
    filtered_data_2D.sort(key=lambda x: x[0])
    filtered_data_selectivity.sort(key=lambda x: x[0])
    
    data_1D_distances = [data[1] for data in filtered_data_1D]
    data_2D_distances = [data[1] for data in filtered_data_2D]
    selectivity_distances = [data[1] for data in filtered_data_selectivity]
    data_names = [data[0] for data in filtered_data_1D]
   
    ax3.scatter(data_1D_distances, selectivity_distances, color='blue', label='filtered_data_1D')
    for x, y, name in zip(data_1D_distances, selectivity_distances, data_names):
        ax3.text(x, y, name, fontsize=8, ha='left', va='top')

    ax4.scatter(data_2D_distances, selectivity_distances, color='red', label='filtered_data_2D')
    for x, y, name in zip(data_2D_distances, selectivity_distances, data_names):
        ax4.text(x, y, name, fontsize=8, ha='left', va='top')

    
    ax3.set_xlabel('Distance')
    ax3.set_ylabel('Binding Selectivity')
    ax3.set_title('Distance vs. Binding Selectivity')
    ax3.legend()
    ax4.set_xlabel('Distance')
    ax4.set_ylabel('Binding Selectivity')
    ax4.set_title('Distance vs. Binding Selectivity')
    ax4.legend()
  
    return 

def KL_divergence_2D(dataset, signal_receptor, target_cells, ax):
    weightDF = convFactCalc()
    target_cells_df, off_target_cells_df, non_signal_receptors, conversion_factor_sig = common_code(weightDF, dataset, signal_receptor, target_cells)
    results = []
    for receptor_name in non_signal_receptors:
        target_receptor_counts = target_cells_df[[signal_receptor, receptor_name]].values
        off_target_receptor_counts = off_target_cells_df[[signal_receptor, receptor_name]].values

    results = []
    for receptor_name in non_signal_receptors:
        target_receptor_counts = target_cells_df[[signal_receptor, receptor_name]].values
        off_target_receptor_counts = off_target_cells_df[[signal_receptor, receptor_name]].values
        conversion_factor = get_conversion_factor(weightDF, receptor_name)
        
        target_receptor_counts[:, 0] *= conversion_factor_sig
        off_target_receptor_counts[:, 0] *= conversion_factor_sig

        target_receptor_counts[:, 1] *= conversion_factor
        off_target_receptor_counts[:, 1] *= conversion_factor

        KL_div = calculate_kl_divergence_2D(target_receptor_counts[:, 0:2], off_target_receptor_counts[:, 0:2])
        if np.mean(target_receptor_counts[:, 0]) > np.mean(off_target_receptor_counts[:, 0]) and np.mean(target_receptor_counts[:, 1]) > np.mean(off_target_receptor_counts[:, 1]):
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

# NOTE: SHOULD IT BE A FUNCTION OR A FIGURE?
def plot_kl_divergence_curves(dataset, signal_receptor, special_receptor, target_cells, ax):
    weightDF = convFactCalc()
    
    IL2Rb_factor = weightDF.loc[weightDF['Receptor'] == 'IL2Rb', 'Weight'].values[0]
    IL7Ra_factor = weightDF.loc[weightDF['Receptor'] == 'IL7Ra', 'Weight'].values[0]
    IL2Ra_factor = weightDF.loc[weightDF['Receptor'] == 'IL2Ra', 'Weight'].values[0]

    target_cells_df = dataset[(dataset['CellType3'] == target_cells) | (dataset['CellType2'] == target_cells)]
    off_target_cells_df = dataset[~((dataset['CellType3'] == target_cells) | (dataset['CellType2'] == target_cells))]

    if signal_receptor == 'CD122':
        conversion_factor_sig = IL2Rb_factor
    elif signal_receptor == 'CD25':
        conversion_factor_sig = IL2Ra_factor
    elif signal_receptor == 'CD127':
        conversion_factor_sig = IL7Ra_factor
    else:
        conversion_factor_sig = (IL7Ra_factor + IL2Ra_factor + IL2Rb_factor) / 3
    
    target_receptor_counts = target_cells_df[special_receptor].values * conversion_factor_sig
    off_target_receptor_counts = off_target_cells_df[special_receptor].values * conversion_factor_sig

    avg_special_receptor = np.mean(np.concatenate([target_receptor_counts, off_target_receptor_counts]))
    target_receptor_counts /= avg_special_receptor
    off_target_receptor_counts /= avg_special_receptor
    
    target_counts_density = stats.gaussian_kde(target_receptor_counts)
    off_target_counts_density = stats.gaussian_kde(off_target_receptor_counts)

    x_vals = np.linspace(min(target_receptor_counts.min(), off_target_receptor_counts.min()),
                         max(target_receptor_counts.max(), off_target_receptor_counts.max()), 1000)

    target_density_curve = target_counts_density(x_vals)
    off_target_density_curve = off_target_counts_density(x_vals)

    ax.plot(x_vals, target_density_curve, color='red', label='Target Cells')
    ax.fill_between(x_vals, target_density_curve, color='red', alpha=0.3)
    
    ax.plot(x_vals, off_target_density_curve, color='blue', label='Off Target Cells')
    ax.fill_between(x_vals, off_target_density_curve, color='blue', alpha=0.3)
    
    ax.set_xscale('log')  # Set x-axis to log scale
    ax.set_xlabel('Receptor Value')
    ax.set_ylabel('Density')
    ax.set_title(f'Receptor Density Curves: {special_receptor} & {signal_receptor}')
    ax.legend()

    KL_div = calculate_kl_divergence_2D(target_receptor_counts, off_target_receptor_counts)
    print(f'KL Divergence for {special_receptor}: {KL_div:.4f}')

# NOTE: SHOULD IT BE A FUNCTION OR A FIGURE?
def plot_2d_density_visualization(dataset, receptor1, receptor2, target_cells, ax):
    target_cells_df = dataset[(dataset['CellType3'] == target_cells) | (dataset['CellType2'] == target_cells)]
    off_target_cells_df = dataset[~((dataset['CellType3'] == target_cells) | (dataset['CellType2'] == target_cells))]

    receptor1_values_target = target_cells_df[receptor1].values
    receptor2_values_target = target_cells_df[receptor2].values

    receptor1_values_off_target = off_target_cells_df[receptor1].values
    receptor2_values_off_target = off_target_cells_df[receptor2].values

    target_kde = KernelDensity(kernel='gaussian').fit(np.column_stack([receptor1_values_target, receptor2_values_target]))
    off_target_kde = KernelDensity(kernel='gaussian').fit(np.column_stack([receptor1_values_off_target, receptor2_values_off_target]))

    # Calculate the KL divergence using the actual receptor values
    KL_div = calculate_kl_divergence_2D(receptor2_values_target, receptor2_values_off_target)
    print(f'KL Divergence for Receptors {receptor1} & {receptor2}: {KL_div:.4f}')

    x_vals = np.linspace(min(receptor1_values_target.min(), receptor1_values_off_target.min()),
                         max(receptor1_values_target.max(), receptor1_values_off_target.max()), 100)
    y_vals = np.linspace(min(receptor2_values_target.min(), receptor2_values_off_target.min()),
                         max(receptor2_values_target.max(), receptor2_values_off_target.max()), 100)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)
    positions = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
    z_target = np.exp(target_kde.score_samples(positions)).reshape(x_grid.shape)
    z_off_target = np.exp(off_target_kde.score_samples(positions)).reshape(x_grid.shape)

    ax.contourf(x_grid, y_grid, z_target, cmap='Reds', alpha=0.5)
    ax.contourf(x_grid, y_grid, z_off_target, cmap='Blues', alpha=0.5)
    
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 125)
    
    ax.set_xlabel(f'Receptor {receptor1}')
    ax.set_ylabel(f'Receptor {receptor2}')
    ax.set_title(f'2D Receptor Density Visualization')
    ax.legend(['Target Cells', 'Off Target Cells'])

# NOTE: SHOULD IT BE A FUNCTION OR A FIGURE?
def bindingmodel_selectivity_pair(dataset, target_cells, signal_receptor, special_receptor):
    cell_types = set(dataset['CellType1']).union(dataset['CellType2']).union(dataset['CellType3'])
    offtarg_cell_types = [cell_type for cell_type in cell_types if cell_type != target_cells]
    epitopes = [column for column in dataset.columns if column not in ['CellType1', 'CellType2', 'CellType3']]
    epitopesDF = getSampleAbundances(epitopes, cell_types, "CellType2") 
    
    dose = 1
    valency = 2

    print(epitopesDF.shape)
    print(epitopesDF.values[1, 1])
    optParams1 = optimizeDesign(signal_receptor, special_receptor, target_cells, offtarg_cell_types, epitopesDF, dose, valency)
    selectivity = 1/optParams1[0]
    return selectivity

def correlation(cell_type, relevant_epitopes):
    epitopesList = pd.read_csv(join(path_here, "data/epitopeList.csv"))
    epitopes = list(epitopesList['Epitope'].unique())
    epitopesDF = getSampleAbundances(epitopes, np.array([cell_type]))

    df = pd.DataFrame()
    for index, row in epitopesDF.iterrows():
        df[row['Epitope']] = row[cell_type]

    corr = df.corr().loc[:, relevant_epitopes]

    return corr