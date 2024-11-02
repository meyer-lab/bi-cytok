import pandas as pd
import seaborn as sns
from .common import getSetup
from ..distanceMetricFuncs import KL_divergence_2D
from ..imports import importCITE


def makeFigure():
    """clustermaps of KL values for receptors + specified cell type"""
    CITE_DF = importCITE()
    CITE_DF = CITE_DF.head(1000)
    
    ax, f = getSetup((40, 40), (1, 1))
    
    targCell = "Treg"
    offTargState = 0

    # Define non-marker columns
    non_marker_columns = ["CellType1", "CellType2", "CellType3", "Cell"]
    marker_columns = CITE_DF.columns[~CITE_DF.columns.isin(non_marker_columns)]
    markerDF = CITE_DF.loc[:, marker_columns]

    # Further filter to include only columns related to CD25 and CD35
    receptors_of_interest = ["CD25", "CD35"]
    filtered_markerDF = markerDF.loc[:, markerDF.columns.str.contains('|'.join(receptors_of_interest), case=False)]

    # Binary arrays for on-target and off-target cell types
    on_target = (CITE_DF["CellType3"] == targCell).astype(int)

    # Define off-target conditions using a dictionary
    off_target_conditions = {
        0: (CITE_DF["CellType3"] != targCell),     # All non-memory Tregs
        1: (CITE_DF["CellType2"] != "Treg"),       # All non-Tregs
        2: (CITE_DF["CellType3"] == "Treg Naive")  # Naive Tregs
    }

    # Set off_target based on offTargState
    if offTargState in off_target_conditions:
        off_target = off_target_conditions[offTargState].astype(int)
    else:
        raise ValueError("Invalid offTargState value. Must be 0, 1, or 2.")
    
    kl_matrix = KL_divergence_2D(filtered_markerDF, on_target, off_target)
    
    df_recep = pd.DataFrame(kl_matrix, index=receptors_of_interest, columns=receptors_of_interest)

    # Visualize with a clustermap
    f = sns.clustermap(df_recep, cmap="bwr", figsize=(10, 10), annot=True, annot_kws={"fontsize": 16})
        
    

    return f
