"""
This file contains functions that are used in multiple figures.
"""
import sys
import time
from string import ascii_lowercase
import seaborn as sns
import numpy as np
import matplotlib
from matplotlib import gridspec, pyplot as plt
import pandas as pd
from ..imports import importCITE
from sklearn.neighbors import KernelDensity
from scipy import stats
import ot
import ot.plot
from ..selectivityFuncs import convFactCalc
from ..selectivityFuncs import convFactCalc
from ..selectivityFuncs import getSampleAbundances, optimizeDesign
from scipy.stats import norm
from scipy.cluster import hierarchy

matplotlib.rcParams["legend.labelspacing"] = 0.2
matplotlib.rcParams["legend.fontsize"] = 8
matplotlib.rcParams["xtick.major.pad"] = 2
matplotlib.rcParams["ytick.major.pad"] = 2
matplotlib.rcParams["xtick.minor.pad"] = 1.9
matplotlib.rcParams["ytick.minor.pad"] = 1.9
matplotlib.rcParams["legend.handletextpad"] = 0.5
matplotlib.rcParams["legend.handlelength"] = 0.5
matplotlib.rcParams["legend.framealpha"] = 0.5
matplotlib.rcParams["legend.markerscale"] = 0.7
matplotlib.rcParams["legend.borderpad"] = 0.35


dosemat = np.array([84, 28, 9.333333, 3.111, 1.037037, 0.345679, 0.115226, 0.038409, 0.012803, 0.004268, 0.001423, 0.000474])


def getSetup(figsize, gridd, multz=None, empts=None):
    """Establish figure set-up with subplots."""
    sns.set(style="whitegrid", font_scale=0.7, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # create empty list if empts isn't specified
    if empts is None:
        empts = []

    if multz is None:
        multz = dict()

    # Setup plotting space and grid
    f = plt.figure(figsize=figsize, constrained_layout=True)
    gs1 = gridspec.GridSpec(*gridd, figure=f)

    # Get list of axis objects
    x = 0
    ax = list()
    while x < gridd[0] * gridd[1]:
        if x not in empts and x not in multz.keys():  # If this is just a normal subplot
            ax.append(f.add_subplot(gs1[x]))
        elif x in multz.keys():  # If this is a subplot that spans grid elements
            ax.append(f.add_subplot(gs1[x: x + multz[x] + 1]))
            x += multz[x]
        x += 1

    return (ax, f)


def genFigure():
    """Main figure generation function."""
    fdir = "./output/"
    start = time.time()
    nameOut = "figure" + sys.argv[1]

    exec("from bicytok.figures." + nameOut + " import makeFigure", globals())
    ff = makeFigure()
    ff.savefig(fdir + nameOut + ".svg", dpi=ff.dpi, bbox_inches="tight", pad_inches=0)

    print(f"Figure {sys.argv[1]} is done after {time.time() - start} seconds.\n")


def subplotLabel(axs):
    """Place subplot labels on figure."""
    for ii, ax in enumerate(axs):
        ax.text(-0.2, 1.25, ascii_lowercase[ii], transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")


cellSTATlimDict = {"Treg": (47000, 54000), "Thelper": (20000, 25000), "CD8": (6200, 7500), "NK": (4000, 5000)}

ratioSTATlimDict = {"Treg/NK": (0, 4000), "Treg/CD8": (0, 1500)}

# ratioSTATlimDict = {"Treg/NK": (0.05, 4),
# "Treg/CD8": (0.05, 3.5)}


def plotBispecific(ax, df, cellType, val=False):
    """Plots all experimental vs. Predicted Values"""

    data_low = df.loc[(df.Cell == cellType) & (df.Affinity == "Low")]
    data_med = df.loc[(df.Cell == cellType) & (df.Affinity == "Medium")]
    data_high = df.loc[(df.Cell == cellType) & (df.Affinity == "High")]

    sns.lineplot(x="Abundance", y="Predicted", data=data_low, label="Low(1e6)", ax=ax, legend="brief")
    sns.lineplot(x="Abundance", y="Predicted", data=data_med, label="Med(1e8)", ax=ax, legend="brief")
    sns.lineplot(x="Abundance", y="Predicted", data=data_high, label="High(1e10)", ax=ax, legend="brief")
    ax.set(title=cellType + " - Dosed at 1nM", xlabel=r"Epitope X Abundance", ylabel="pSTAT", xscale="log", ylim=cellSTATlimDict[cellType])


def Wass_KL_Dist(ax, targCell, numFactors, RNA=False, offTargState=0):
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

def EMD_Distribution_Plot(ax, dataset, signal_receptor, non_signal_receptor, target_cells):
    target_cells_df = dataset[(dataset['CellType3'] == target_cells) | (dataset['CellType2'] == target_cells)]
    off_target_cells_df = dataset[~((dataset['CellType3'] == target_cells) | (dataset['CellType2'] == target_cells))]
    
    xs = target_cells_df[[signal_receptor, non_signal_receptor]].values
    xt = off_target_cells_df[[signal_receptor, non_signal_receptor]].values

    M = ot.dist(xs, xt)
    a = np.ones((xs.shape[0],)) / xs.shape[0]
    b = np.ones((xt.shape[0],)) / xt.shape[0]
  
    G0 = ot.emd2(a, b, M, numItermax=10000000)

    ax.plot(xt[:, 0], xt[:, 1], '.r', label='Off-target cells')
    ax.plot(xs[:, 0], xs[:, 1], '.b', label='Target cells')
    
    ax.legend(loc=0)
    ax.set_title('Target cell and off-target cell distributions')

    ax.set_xlabel(signal_receptor)
    ax.set_ylabel(non_signal_receptor)
    
    ax.set_xscale('log')  # Set x-axis to logarithmic scale
    ax.set_yscale('log')
    ax.legend(loc=0, fontsize=12)

    return

def EMD_2D(dataset, signal_receptor, target_cells, ax):
    weightDF = convFactCalc()
    # target and off-target cells
    IL2Rb_factor = weightDF.loc[weightDF['Receptor'] == 'IL2Rb', 'Weight'].values[0]
    IL7Ra_factor = weightDF.loc[weightDF['Receptor'] == 'IL7Ra', 'Weight'].values[0]
    IL2Ra_factor = weightDF.loc[weightDF['Receptor'] == 'IL2Ra', 'Weight'].values[0]
    
    non_signal_receptors = []
    for column in dataset.columns:
        if column != signal_receptor and column not in ['CellType1', 'CellType2', 'CellType3']:
            non_signal_receptors.append(column)

    results = []
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
    
    for receptor_name in non_signal_receptors:
        target_receptor_counts = target_cells_df[[signal_receptor, receptor_name]].values
        off_target_receptor_counts = off_target_cells_df[[signal_receptor, receptor_name]].values

        if receptor_name == 'CD122':
            conversion_factor = IL2Rb_factor
        elif receptor_name == 'CD25':
            conversion_factor = IL2Ra_factor
        elif receptor_name == 'CD127':
            conversion_factor = IL7Ra_factor
        else:
            conversion_factor = (IL7Ra_factor+IL2Ra_factor+IL2Rb_factor)/3

        target_receptor_counts[:, 0] *= conversion_factor_sig
        off_target_receptor_counts[:, 0] *= conversion_factor_sig

        target_receptor_counts[:, 1] *= conversion_factor
        off_target_receptor_counts[:, 1] *= conversion_factor
        
        average_receptor_counts = np.mean(np.concatenate((target_receptor_counts, off_target_receptor_counts)))

        # Normalize the counts by dividing by the average
        target_receptor_counts = target_receptor_counts.astype(float) / average_receptor_counts
        off_target_receptor_counts = off_target_receptor_counts.astype(float) / average_receptor_counts
        
        # Matrix for emd parameter
        M = ot.dist(target_receptor_counts, off_target_receptor_counts)
        # optimal transport distance
        a = np.ones((target_receptor_counts.shape[0],)) / target_receptor_counts.shape[0]
        b = np.ones((off_target_receptor_counts.shape[0],)) / off_target_receptor_counts.shape[0]
        optimal_transport = ot.emd2(a, b, M, numItermax=10000000)
        if np.mean(target_receptor_counts[:, 1]) > np.mean(off_target_receptor_counts[:, 1]): 
            results.append((optimal_transport, receptor_name, signal_receptor)) #indent if using if statement
        else:
            results.append((0, receptor_name, signal_receptor))
            
    # end loop
    sorted_results = sorted(results, reverse=True)
    
    top_receptor_info = [(receptor_name, optimal_transport, signal_receptor) for optimal_transport, receptor_name, signal_receptor in sorted_results[:5]]
    
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
        
       
        average_receptor_counts = np.mean(np.concatenate((target_receptor_counts, off_target_receptor_counts)))

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
    prevOptAffs = [8.0, 8.0, 8.0]
    dose = 1
    valency = 2
    
    for receptor_name in receptor_names:
        print("Receptor name:", receptor_name)
        optParams1 = optimizeDesign(signal_receptor, receptor_name, target_cells, offtarg_cell_types, epitopesDF, dose, valency, prevOptAffs)
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

def EMD_3D(dataset, signaling_receptor, target_cells, ax):
    weightDF = convFactCalc()
    # target and off-target cells
    IL2Rb_factor = weightDF.loc[weightDF['Receptor'] == 'IL2Rb', 'Weight'].values[0]
    IL7Ra_factor = weightDF.loc[weightDF['Receptor'] == 'IL7Ra', 'Weight'].values[0]
    IL2Ra_factor = weightDF.loc[weightDF['Receptor'] == 'IL2Ra', 'Weight'].values[0]

    non_signaling_receptors = []
    for column in dataset.columns:
        if column != signaling_receptor and column !='CD25' and column not in ['CellType1', 'CellType2', 'CellType3']: # must specify cd25 or other 3rd receptor
            non_signaling_receptors.append(column)

    results = []
    target_cells_df = dataset[(dataset['CellType3'] == target_cells) | (dataset['CellType2'] == target_cells)]
    off_target_cells_df = dataset[~((dataset['CellType3'] == target_cells) | (dataset['CellType2'] == target_cells))]

    if signaling_receptor == 'CD122':
        conversion_factor_sig = IL2Rb_factor
    elif signaling_receptor == 'CD25':
        conversion_factor_sig = IL2Ra_factor
    elif signaling_receptor == 'CD127':
        conversion_factor_sig = IL7Ra_factor
    else:
        conversion_factor_sig = (IL7Ra_factor + IL2Ra_factor + IL2Rb_factor) / 3

    for receptor_name in non_signaling_receptors:
        target_receptor_counts = target_cells_df[[signaling_receptor, receptor_name, 'CD25']].values
        off_target_receptor_counts = off_target_cells_df[[signaling_receptor, receptor_name, 'CD25']].values

        if receptor_name == 'CD122':
            conversion_factor = IL2Rb_factor
        elif receptor_name == 'CD25':
            conversion_factor = IL2Ra_factor
        elif receptor_name == 'CD127':
            conversion_factor = IL7Ra_factor
        else:
            conversion_factor = (IL7Ra_factor + IL2Ra_factor + IL2Rb_factor) / 3

        target_receptor_counts[:, 0] *= conversion_factor_sig
        off_target_receptor_counts[:, 0] *= conversion_factor_sig

        target_receptor_counts[:, 1] *= conversion_factor
        off_target_receptor_counts[:, 1] *= conversion_factor

        target_receptor_counts[:, 2] *= conversion_factor 
        off_target_receptor_counts[:, 2] *= conversion_factor

        average_receptor_counts = np.mean(np.concatenate((target_receptor_counts, off_target_receptor_counts)))

        # Normalize the counts by dividing by the average
        target_receptor_counts = target_receptor_counts.astype(float) / average_receptor_counts
        off_target_receptor_counts = off_target_receptor_counts.astype(float) / average_receptor_counts

        # Matrix for emd parameter
        M = ot.dist(target_receptor_counts, off_target_receptor_counts)
        # optimal transport distance
        a = np.ones((target_receptor_counts.shape[0],)) / target_receptor_counts.shape[0]
        b = np.ones((off_target_receptor_counts.shape[0],)) / off_target_receptor_counts.shape[0]

        optimal_transport = ot.emd2(a, b, M, numItermax=10000000)
        if np.mean(target_receptor_counts[:, 1]) > np.mean(off_target_receptor_counts[:, 1]):
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
    ax.set_title('Top 5 Receptor Distances (3D)')
    ax.set_xticks(range(len(receptor_names)))
    ax.set_xticklabels(receptor_names, rotation='vertical')

    print('The 5 non-signaling receptors that achieve the greatest positive distance from target-off-target cells are:', top_receptor_info)
    return sorted_results

def calculate_kl_divergence_2D(targCellMark, offTargCellMark):
    kdeTarg = KernelDensity(kernel='gaussian').fit(targCellMark.reshape(-1, 1))
    kdeOffTarg = KernelDensity(kernel='gaussian').fit(offTargCellMark.reshape(-1, 1))
    minVal = np.minimum(targCellMark.min(), offTargCellMark.min()) - 10
    maxVal = np.maximum(targCellMark.max(), offTargCellMark.max()) + 10
    outcomes = np.arange(minVal, maxVal + 1).reshape(-1, 1)
    distTarg = np.exp(kdeTarg.score_samples(outcomes))
    distOffTarg = np.exp(kdeOffTarg.score_samples(outcomes))
    KL_div = stats.entropy(distOffTarg.flatten() + 1e-200, distTarg.flatten() + 1e-200, base=2)
    return KL_div

def KL_divergence_2D(dataset, signal_receptor, target_cells, ax):
    weightDF = convFactCalc()
    
    IL2Rb_factor = weightDF.loc[weightDF['Receptor'] == 'IL2Rb', 'Weight'].values[0]
    IL7Ra_factor = weightDF.loc[weightDF['Receptor'] == 'IL7Ra', 'Weight'].values[0]
    IL2Ra_factor = weightDF.loc[weightDF['Receptor'] == 'IL2Ra', 'Weight'].values[0]

    non_signal_receptors = []
    for column in dataset.columns:
        if column != signal_receptor and column not in ['CellType1', 'CellType2', 'CellType3', 'Cell']:
            non_signal_receptors.append(column)

    results = []
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
    
    for receptor_name in non_signal_receptors:
        target_receptor_counts = target_cells_df[[signal_receptor, receptor_name]].values
        off_target_receptor_counts = off_target_cells_df[[signal_receptor, receptor_name]].values

        if receptor_name == 'CD122':
            conversion_factor = IL2Rb_factor
        elif receptor_name == 'CD25':
            conversion_factor = IL2Ra_factor
        elif receptor_name == 'CD127':
            conversion_factor = IL7Ra_factor
        else:
            conversion_factor = (IL7Ra_factor + IL2Ra_factor + IL2Rb_factor) / 3

        target_receptor_counts[:, 0] *= conversion_factor_sig
        off_target_receptor_counts[:, 0] *= conversion_factor_sig

        target_receptor_counts[:, 1] *= conversion_factor
        off_target_receptor_counts[:, 1] *= conversion_factor
        
        KL_div = calculate_kl_divergence_2D(target_receptor_counts[:, 0:2], off_target_receptor_counts[:, 0:2])
        if np.mean(target_receptor_counts[:, 1]) > np.mean(off_target_receptor_counts[:, 1]): 
            results.append((KL_div, receptor_name, signal_receptor))
        else:
            results.append((-1, receptor_name, signal_receptor))
    
    sorted_results = sorted(results, reverse=True)
    top_receptor_info = [(receptor_name, KL_div, signal_receptor) for KL_div, receptor_name, signal_receptor in sorted_results[:5]]
    
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

#################################################
def KL_divergence_forheatmap(dataset, signal_receptor, target_cells):
    weightDF = convFactCalc()
    
    IL2Rb_factor = weightDF.loc[weightDF['Receptor'] == 'IL2Rb', 'Weight'].values[0]
    IL7Ra_factor = weightDF.loc[weightDF['Receptor'] == 'IL7Ra', 'Weight'].values[0]
    IL2Ra_factor = weightDF.loc[weightDF['Receptor'] == 'IL2Ra', 'Weight'].values[0]

    non_signal_receptors = []
    for column in dataset.columns:
        if column != signal_receptor and column not in ['CellType1', 'CellType2', 'CellType3', 'Cell']:
            non_signal_receptors.append(column)

    results = []
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
    
    for receptor_name in non_signal_receptors:
        target_receptor_counts = target_cells_df[[signal_receptor, receptor_name]].values
        off_target_receptor_counts = off_target_cells_df[[signal_receptor, receptor_name]].values

        if receptor_name == 'CD122':
            conversion_factor = IL2Rb_factor
        elif receptor_name == 'CD25':
            conversion_factor = IL2Ra_factor
        elif receptor_name == 'CD127':
            conversion_factor = IL7Ra_factor
        else:
            conversion_factor = (IL7Ra_factor + IL2Ra_factor + IL2Rb_factor) / 3

        target_receptor_counts[:, 0] *= conversion_factor_sig
        off_target_receptor_counts[:, 0] *= conversion_factor_sig

        target_receptor_counts[:, 1] *= conversion_factor
        off_target_receptor_counts[:, 1] *= conversion_factor

        KL_div = calculate_kl_divergence_2D(target_receptor_counts[:, 1], off_target_receptor_counts[:, 1])
        results.append((KL_div, receptor_name))
       
     
    all_receptor_info = [(receptor_name, KL_div) for KL_div, receptor_name in results] 
  
    return all_receptor_info

    

#################################################

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
        
    average_receptor_counts = np.mean(np.concatenate((target_receptor_counts, off_target_receptor_counts)))

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
        
    average_receptor_counts = np.mean(np.concatenate((target_receptor_counts, off_target_receptor_counts)))

    # Normalize the counts by dividing by the average
    target_receptor_counts = target_receptor_counts.astype(float) / average_receptor_counts
    off_target_receptor_counts = off_target_receptor_counts.astype(float) / average_receptor_counts
        
    KL_div = calculate_kl_divergence_2D(target_receptor_counts[:, 1], off_target_receptor_counts[:, 1])
    
    print('KL', KL_div)
    return KL_div

def bindingmodel_selectivity_pair(dataset, target_cells, signal_receptor, special_receptor):
    cell_types = set(dataset['CellType1']).union(dataset['CellType2']).union(dataset['CellType3'])
    offtarg_cell_types = [cell_type for cell_type in cell_types if cell_type != target_cells]
    epitopes = [column for column in dataset.columns if column not in ['CellType1', 'CellType2', 'CellType3']]
    epitopesDF = getSampleAbundances(epitopes, cell_types, "CellType2") 

    # Set the appropriate number of affinity parameters based on the signal_receptor
    prevOptAffs = [8.0, 8.0, 8.0] if signal_receptor == 'CD122' else [8.0, 8.0]
    
    dose = 1
    valency = 2

    print(epitopesDF.shape)
    print(epitopesDF.values[1, 1])
    optParams1 = optimizeDesign(signal_receptor, special_receptor, target_cells, offtarg_cell_types, epitopesDF, dose, valency, prevOptAffs)
    selectivity = 1/optParams1[0]
    return selectivity