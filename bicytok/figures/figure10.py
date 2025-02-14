"""
Figure serves to visualize the variability of the distance metrics and selectivity
as it varies with sample size.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..distance_metric_funcs import KL_EMD_2D
from ..imports import importCITE
from ..selectivity_funcs import optimize_affs, sample_receptor_abundances
from .common import getSetup

path_here = Path(__file__).parent.parent


def makeFigure():
    ax, f = getSetup((15, 5), (1, 3))

    # Parameters
    sample_sizes = [50, 100, 200, 500, 1000, 2000, 5000]
    randomizations = 100

    targCell = "Treg"
    signal_receptor = "CD122"
    valencies = np.array([[1, 1, 1]])
    targets = ["CD25", "CD278"]
    dose = 10e-2
    cellTypes = np.array(
        [
            "CD8 Naive",
            "NK",
            "CD8 TEM",
            "CD4 Naive",
            "CD4 CTL",
            "CD8 TCM",
            "CD8 Proliferating",
            "Treg",
        ]
    )
    offTargCells = cellTypes[cellTypes != targCell]
    cell_categorization = "CellType2"

    # Imports
    CITE_DF = importCITE()

    assert targCell in CITE_DF[cell_categorization].unique()

    epitopesDF = CITE_DF[[signal_receptor] + targets + [cell_categorization]]
    epitopesDF = epitopesDF.loc[epitopesDF[cell_categorization].isin(cellTypes)]
    epitopesDF = epitopesDF.rename(columns={cell_categorization: "Cell Type"})
    sampleDF = sample_receptor_abundances(
        CITE_DF=epitopesDF,
        numCells=epitopesDF.shape[0],
        targCellType=targCell,
        offTargCellTypes=offTargCells,
    )

    metrics = []
    for sample_size in sample_sizes:
        target_cells = sampleDF[sampleDF["Cell Type"] == targCell]
        off_target_cells = sampleDF[sampleDF["Cell Type"].isin(offTargCells)]

        num_target_cells = sample_size // 2
        num_off_target_cells = sample_size - num_target_cells

        # Bootstrap
        for _ in range(randomizations):
            # Randomly sample a subset of rows
            sampled_target_cells = target_cells.sample(
                min(num_target_cells, target_cells.shape[0]), replace=True
            )
            sampled_off_target_cells = off_target_cells.sample(
                min(num_off_target_cells, off_target_cells.shape[0]), replace=True
            )
            rand_samples = pd.concat([sampled_target_cells, sampled_off_target_cells])

            # Define target and off-target cell masks (for distance metrics)
            target_mask = (rand_samples["Cell Type"] == targCell).to_numpy()
            off_target_mask = rand_samples["Cell Type"].isin(offTargCells).to_numpy()

            # Calculate distance metrics
            rec_abundances = rand_samples[targets].to_numpy()
            KL_div_mat, EMD_mat = KL_EMD_2D(
                rec_abundances, target_mask, off_target_mask, calc_1D=False
            )
            KL_div = KL_div_mat[1, 0]
            EMD = EMD_mat[1, 0]

            # Define target and off-target cell dataframes (for model)
            targ_df = rand_samples.loc[target_mask]
            off_targ_df = rand_samples.loc[off_target_mask]
            targ_recs = targ_df[[signal_receptor] + targets].to_numpy()
            off_targ_recs = off_targ_df[[signal_receptor] + targets].to_numpy()

            # Calculate selectivity and affinities
            opt_selec, opt_affs = optimize_affs(
                targRecs=targ_recs,
                offTargRecs=off_targ_recs,
                dose=dose,
                valencies=valencies,
            )

            metrics.append(
                {
                    "sample_size": sample_size,
                    "KL_div": KL_div,
                    "EMD": EMD,
                    "selectivity": 1 / opt_selec,
                    "affinities": opt_affs,
                }
            )

    metrics_df = pd.DataFrame(metrics)

    # Plotting
    sns.boxplot(x="sample_size", y="KL_div", data=metrics_df, ax=ax[0])
    ax[0].set_title("KL Divergence")
    ax[0].set_xlabel("Sample Size")
    ax[0].set_ylabel("KL Divergence")

    sns.boxplot(x="sample_size", y="EMD", data=metrics_df, ax=ax[1])
    ax[1].set_title("Earth Mover's Distance")
    ax[1].set_xlabel("Sample Size")
    ax[1].set_ylabel("EMD")

    sns.boxplot(x="sample_size", y="selectivity", data=metrics_df, ax=ax[2])
    ax[2].set_title("Selectivity")
    ax[2].set_xlabel("Sample Size")
    ax[2].set_ylabel("Selectivity")

    plt.tight_layout()

    return f
