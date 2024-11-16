import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..distanceMetricFuncs import EMD_2D
from ..imports import importCITE
from .common import getSetup


def makeFigure():
    '''
    Generates a heatmap visualizing the Earth Mover's Distance (EMD) between selected receptors (CD25 and CD35) for a target cell type ("Treg") compared to off-target populations
    
    Data Import:
   - Loads the CITE-seq dataset using `importCITE` and samples the first 1000 rows for analysis.
   - Identifies non-marker columns (`CellType1`, `CellType2`, `CellType3`, `Cell`) and filters out these columns 
     to retain only the marker (receptor) columns for analysis.

    Receptor Selection:
   - Filters the marker dataframe to include only columns related to the receptors of interest, specifically `"CD25"` and `"CD35"`, 
     focusing on these markers for the calculation of EMD.

    Target and Off-Target Cell Definition:
   - Creates a binary array for on-target cells based on the specified target cell type (`"Treg"`).
   - Defines off-target cell populations using the `offTargState` parameter:
     - `offTargState = 0`: All non-memory Tregs.
     - `offTargState = 1`: All non-Tregs.
     - `offTargState = 2`: Only naive Tregs.

    EMD Calculation:
   - Computes an Earth Mover's Distance (EMD) matrix using the `EMD_2D` function to measure the dissimilarity 
     between on-target ("Treg") and off-target cell distributions for the selected receptors (CD25 and CD35).
   - Constructs a DataFrame (`df_recep`) to store the computed EMD values, with rows and columns labeled by the receptors of interest.

    Visualization:
   - Generates a heatmap of the EMD matrix using Seaborn's `heatmap` function.
   - The heatmap uses a "bwr" color map to visually represent the EMD values, with annotations to display specific values.
    '''
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
    filtered_markerDF = markerDF.loc[
        :, markerDF.columns.str.contains("|".join(receptors_of_interest), case=False)
    ]

    # Binary arrays for on-target and off-target cell types
    on_target = (CITE_DF["CellType3"] == targCell).astype(int)

    # Define off-target conditions using a dictionary
    off_target_conditions = {
        0: (CITE_DF["CellType3"] != targCell),  # All non-memory Tregs
        1: (CITE_DF["CellType2"] != "Treg"),  # All non-Tregs
        2: (CITE_DF["CellType3"] == "Treg Naive"),  # Naive Tregs
    }

    if offTargState in off_target_conditions:
        off_target_mask = off_target_conditions[offTargState]
    else:
        raise ValueError("Invalid offTargState value. Must be 0, 1, or 2.")

    on_target_values = filtered_markerDF[on_target.astype(bool)].values
    off_target_values = filtered_markerDF[off_target_mask].values

    EMD_matrix = EMD_2D(on_target_values, off_target_values)

    df_recep = pd.DataFrame(
        EMD_matrix, index=receptors_of_interest, columns=receptors_of_interest
    )

    # Visualize with a clustermap
    sns.heatmap(
        df_recep, cmap="bwr", annot=True, ax=ax, cbar=True, annot_kws={"fontsize": 16}
    )

    ax.set_title("EMD between: CD25 and CD35")
    plt.show()

    return f
