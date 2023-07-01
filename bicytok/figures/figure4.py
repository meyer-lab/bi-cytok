
from ot.datasets import make_1D_gauss as gauss
from .common import getSetup
from .common import Wass_KL_Dist2d
from ..imports import importCITE
import matplotlib.pyplot as plt
from ot import emd2_samples




def makeFigure():
    f, ax = getSetup((10, 8), (1, 1))
    targCell = "Treg"
    numFactors = 5
    offTargReceptors = ["CD335"]  # Update with the list of off-target receptors
    signalReceptor = "CD122"  # Update with the signaling receptor
    
    Wass_KL_Dist2d(ax, targCell, numFactors, offTargReceptors, signalReceptor)

    # Display the bar plots
    plt.show()
    return f
