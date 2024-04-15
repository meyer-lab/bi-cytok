from .common import getSetup, Wass_KL_Dist


def makeFigure():
    ax, f = getSetup((8, 8), (3, 2))

    Wass_KL_Dist(ax[0:2], "Treg Memory", 10)
    Wass_KL_Dist(ax[2:4], "Treg Memory", 10, offTargState=1)
    Wass_KL_Dist(ax[4:6], "Treg Memory", 10, offTargState=2)
  
    return f
