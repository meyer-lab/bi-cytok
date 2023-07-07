from os.path import dirname, join
from .common import getSetup
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from .common import EMD_Receptors
from .common import OT_Matrix_Plot
from ..imports import importCITE


path_here = dirname(dirname(__file__))

def makeFigure():
    markerDF = importCITE()
    new_df = markerDF.head(1000)
    f, ax = plt.subplots(1, 2, figsize=(20, 10))
    target_cells = 'Treg'
    signaling_receptor = 'CD122'
    non_siganling_receptor = 'CD4-1'
    EMD_Receptors(new_df, signaling_receptor, target_cells, ax[0])
    OT_Matrix_Plot(ax[1], new_df, signaling_receptor, non_siganling_receptor, target_cells)

    
    return f
