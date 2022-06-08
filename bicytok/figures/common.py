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
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend([handles[0]] + handles[4::], [labels[0]] + labels[4::])
