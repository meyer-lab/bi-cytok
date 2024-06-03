from os.path import dirname, join

import numpy as np
import pandas as pd
import seaborn as sns

from ..distanceMetricFuncs import EMD_2D, KL_divergence_2D, correlation
from ..imports import importCITE
from ..selectivityFuncs import calcReceptorAbundances, optimizeSelectivityAffs
from .common import getSetup

path_here = dirname(dirname(__file__))


SIGNAL_RECEPTOR = "CD122"
SIGNAL_VALENCY = 1
VALENCIES = [1, 2, 4]
ALL_TARGETS = [["CD25", "CD278"], ["CD25", "CD4-2"], ["CD25", "CD45RB"]]
DOSE = 10e-2
CELLS = np.array(
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
TARG_CELL = "Treg"

def makeFigure():
    """Figure file to generate plots of bispecific ligand selectivity for combinations of different KL divergences, EMDs, and anti-correlations."""
    ax, f = getSetup((9, 3), (1, 3))

    CITE_DF = importCITE()
    new_df = CITE_DF.sample(1000, random_state=42)

    offTCells = CELLS[CELLS != TARG_CELL]

    epitopesList = pd.read_csv(join(path_here, "data/epitopeList.csv"))
    epitopes = list(epitopesList["Epitope"].unique())
    epitopesDF = calcReceptorAbundances(epitopes, CELLS, numCells=1000)

    df = pd.DataFrame(
        columns=[
            "KL Divergence",
            "Earth Mover's Distance",
            "Correlation",
            "Selectivity",
            "Valency",
        ]
    )

    for val in VALENCIES:
        prevOptAffs = [8.0, 8.0, 8.0]
        for targets in ALL_TARGETS:
            vals = np.array([[SIGNAL_VALENCY, val, val]])

            optParams = optimizeSelectivityAffs(
                SIGNAL_RECEPTOR,
                targets,
                TARG_CELL,
                offTCells,
                epitopesDF,
                DOSE,
                vals,
                prevOptAffs,
            )
            prevOptAffs = optParams[1]
            select = (1 / optParams[0],)
            KLD = KL_divergence_2D(new_df, targets[0], TARG_CELL, targets[1], ax=None)
            EMD = EMD_2D(new_df, targets[0], TARG_CELL, targets[1], ax=None)
            corr = correlation(TARG_CELL, targets).loc[targets[0], targets[1]][
                "Correlation"
            ]

            data = {
                "KL Divergence": [KLD],
                "Earth Mover's Distance": [EMD],
                "Correlation": [corr],
                "Selectivity": select,
                "Valency": [val],
            }
            df_temp = pd.DataFrame(
                data,
                columns=[
                    "KL Divergence",
                    "Earth Mover's Distance",
                    "Correlation",
                    "Selectivity",
                    "Valency",
                ],
            )
            df = pd.concat([df, df_temp], ignore_index=True)
    sns.lineplot(data=df, x="KL Divergence", y="Selectivity", hue="Valency", ax=ax[0])
    sns.lineplot(
        data=df, x="Earth Mover's Distance", y="Selectivity", hue="Valency", ax=ax[1]
    )
    sns.lineplot(data=df, x="Correlation", y="Selectivity", hue="Valency", ax=ax[2])
    ax[0].set(xscale="log")
    ax[1].set(xscale="log")

    return f
