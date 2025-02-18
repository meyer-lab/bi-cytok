"""Generate plots to compare 1D and 2D distance metrics, which should match."""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from ..distance_metric_funcs import KL_EMD_1D, KL_EMD_2D
from ..imports import importCITE, sample_receptor_abundances
from .common import getSetup


def makeFigure():
    ax, f = getSetup((10, 5), (2, 2))
    np.random.seed(42)

    # Parameters
    targCell = "Treg"
    receptors_of_interest = [
        "CD25",
        "CD4-1",
        "CD27",
        "CD4-2",
        "CD278",
        "CD122",
    ]
    sample_size = 500

    # Define non-marker columns
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

    CITE_DF = importCITE()

    assert targCell in CITE_DF[cell_categorization].unique()

    epitopesDF = CITE_DF[receptors_of_interest + [cell_categorization]]
    epitopesDF = epitopesDF.loc[epitopesDF[cell_categorization].isin(cellTypes)]
    epitopesDF = epitopesDF.rename(columns={cell_categorization: "Cell Type"})
    sampleDF = sample_receptor_abundances(
        CITE_DF=epitopesDF,
        numCells=sample_size,
        targCellType=targCell,
        offTargCellTypes=offTargCells,
    )
    rec_abundances = sampleDF[receptors_of_interest].to_numpy()

    target_mask = sampleDF["Cell Type"] == targCell
    off_target_mask = sampleDF["Cell Type"].isin(offTargCells)

    KL_div_vals_1D, EMD_vals_1D = KL_EMD_1D(
        rec_abundances, target_mask, off_target_mask
    )

    KL_div_vals_2D, EMD_vals_2D = KL_EMD_2D(
        rec_abundances, target_mask, off_target_mask, calc_1D=True
    )
    KL_div_vals_2D = np.diag(KL_div_vals_2D)
    EMD_vals_2D = np.diag(EMD_vals_2D)

    # EMD values have exact linear scaling between 1D and 2D based on Euclidean
    #   distance. KL divergence values have some non-linear scaling. The EMD
    #   scaling factor does a pretty good job of approximating the KL divergence.
    EMD_vals_2D_scaled = EMD_vals_2D * 2**0.5 / 2
    KL_div_vals_2D_scaled = KL_div_vals_2D * 2**0.5 / 2

    KL_div_r2 = 1 - np.sum((KL_div_vals_1D - KL_div_vals_2D_scaled) ** 2) / np.sum(
        (KL_div_vals_1D - np.mean(KL_div_vals_1D)) ** 2
    )
    print(f"R^2 for KL divergence with EMD scaling factor: {KL_div_r2}")

    # KL div polynomial model (best fit, likely overfit)
    poly = PolynomialFeatures(degree=2)
    kl_model = make_pipeline(poly, LinearRegression())
    kl_model.fit(KL_div_vals_2D.reshape(-1, 1), KL_div_vals_1D)
    print(kl_model.score(KL_div_vals_2D.reshape(-1, 1), KL_div_vals_1D))
    print(kl_model.named_steps["linearregression"].intercept_)
    print(kl_model.named_steps["linearregression"].coef_)

    # KL div linear model (better fit than EMD scaling, worse than polynomial)
    kl_model = LinearRegression()
    kl_model.fit(KL_div_vals_2D.reshape(-1, 1), KL_div_vals_1D)
    print(kl_model.score(KL_div_vals_2D.reshape(-1, 1), KL_div_vals_1D))
    print(kl_model.intercept_)
    print(kl_model.coef_)

    # EMD linear model (perfect fit)
    emd_model = LinearRegression()
    emd_model.fit(EMD_vals_2D_scaled.reshape(-1, 1), EMD_vals_1D)
    print(emd_model.score(EMD_vals_2D_scaled.reshape(-1, 1), EMD_vals_1D))
    print(emd_model.intercept_)
    print(emd_model.coef_)

    # Plot KL values
    ax[0].barh(
        receptors_of_interest,
        KL_div_vals_1D,
        color="b",
    )
    ax[0].set_title("1D KL Divergence Values")
    ax[0].set_xlabel("1D KL Divergence")
    ax[2].barh(
        receptors_of_interest,
        KL_div_vals_2D_scaled,
        color="b",
    )
    ax[2].set_title("2D KL Divergence Values")
    ax[2].set_xlabel("2D KL Divergence")

    # Plot EMD values
    ax[1].barh(
        receptors_of_interest,
        EMD_vals_1D,
        color="g",
    )
    ax[1].set_title("1D EMD Values")
    ax[1].set_xlabel("1D EMD")
    ax[3].barh(
        receptors_of_interest,
        EMD_vals_2D_scaled,
        color="g",
    )
    ax[3].set_title("2D EMD Values")
    ax[3].set_xlabel("2D EMD")

    return f
