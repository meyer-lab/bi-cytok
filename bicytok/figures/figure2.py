from .common import getSetup
from ..distanceMetricFuncs import KL_EMD_1D


def makeFigure():
    """1D KL divergence and EMD for given cell type/subset."""
    ax, f = getSetup((8, 8), (3, 2))

    KL_EMD_1D(ax[0:2], "Treg Memory", 10)
    KL_EMD_1D(ax[2:4], "Treg Memory", 10, offTargState=1)
    KL_EMD_1D(ax[4:6], "Treg Memory", 10, offTargState=2)

    return f
