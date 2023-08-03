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
    elif receptor_name == 'CD25':
        conversion_factor_sig = IL2Ra_factor
    elif receptor_name == 'CD127':
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
    ax.set_title('Top 5 Receptor Distances (2D)')
    ax.set_xticks(range(len(receptor_names)))
    ax.set_xticklabels(receptor_names, rotation='vertical')
    
    print('The 5 non signaling receptors which achieve the greatest positive distance from target-off-target cells are:', top_receptor_info)
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

        # Normalize the counts by dividing by the average
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
    dose = .1
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
