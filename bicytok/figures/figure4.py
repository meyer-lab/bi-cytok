from os.path import dirname, join
from .common import getSetup
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from .common import Wass_KL_Dist
from .common import calculate_distance
from ..imports import importCITE


path_here = dirname(dirname(__file__))

def makeFigure():
    markerDF = importCITE()
    new_df = markerDF.head(1000)
    ax, f = getSetup((8, 8), (2, 2)) # works
    target_cells = 'Treg'
    signaling_receptor = 'CD122'
    off_target_receptors = 'CD25'

    top_distances = calculate_distance(new_df, signaling_receptor, off_target_receptors, target_cells)
    print("Top 5 noptimal transport distances, obviously rn im aware it will not give 5 bc i only have 2 receptors:", top_distances)

    return f
