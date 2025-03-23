import seaborn as sns
from pathlib import Path
import numpy as np
import pandas as pd
from ..distance_metric_funcs import KL_EMD_2D, KL_EMD_1D
from ..imports import importCITE, sample_receptor_abundances
from .common import getSetup

path_here = Path(__file__).parent.parent

def makeFigure():
    ax, f = getSetup((14, 14), (1, 2)) 

    receptors_of_interest = [
    "CD25",
    "CD278",
    "CD4-1",
    "CD27",
    "CD45RB",
    "CD28",
    "TCR-2",
    "TIGIT",
    "CD4-2",
    "CD122",
    "CD3-1",
    "CD3-2",
    "CD146"
]

    sample_size = 1000
    targCell = "Treg"
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

    sample_size = 100
    cell_categorization = "CellType2"

    CITE_DF = importCITE()

    assert targCell in CITE_DF[cell_categorization].unique()

    epitopes = [
        col
        for col in CITE_DF.columns
        if col not in ["CellType1", "CellType2", "CellType3"]
    ]
    epitopesDF = CITE_DF[epitopes + [cell_categorization]]
    epitopesDF = epitopesDF.rename(columns={cell_categorization: "Cell Type"})
    sampleDF = sample_receptor_abundances(
        CITE_DF=epitopesDF,
        numCells=min(sample_size, epitopesDF.shape[0]),
        targCellType=targCell,
    )
    
    
    filtered_sampleDF = sampleDF.loc[
        :,
        sampleDF.columns.str.fullmatch("|".join(receptors_of_interest), case=False),
    ]
    raw_sampleDF = epitopesDF.copy()
    filtered_raw_sampleDF = raw_sampleDF.loc[
        sampleDF.index,  
        sampleDF.columns
    ]
    print ("sampleDF", sampleDF)

    print ("raw_sampleDF", raw_sampleDF)
    
    
    print ("filtered_sampleDF", filtered_sampleDF)

    on_target_mask = (sampleDF["Cell Type"] == targCell).to_numpy()
    off_target_mask = ~on_target_mask

    rec_abundances = filtered_sampleDF.to_numpy()
    rec_abundancesraw = filtered_raw_sampleDF.to_numpy()

    KL_div_vals, EMD_vals = KL_EMD_2D(
        rec_abundances, on_target_mask, off_target_mask, calc_1D=False
    )

    KL_div_valsRAW, EMD_valsRAW = KL_EMD_2D(
        rec_abundancesraw, on_target_mask, off_target_mask, calc_1D=False
    )

    print("EMD_vals", EMD_vals)
    print("EMD_valsRAW", EMD_valsRAW)
    print ("KL_div_vals", KL_div_vals)
    print ("KL_div_valsRAW", KL_div_valsRAW)
    metrics_raw = {
        "Receptor Pair": receptors_of_interest,
        "Data Type": ["Raw"] * len(receptors_of_interest),
        "KL Divergence": KL_div_valsRAW,
        "EMD": EMD_valsRAW
    }
    metrics_df2_raw = pd.DataFrame(metrics_raw)

    # Prepare the abundance metrics DataFrame
    metrics_abundance = {
        "Receptor Pair": receptors_of_interest,
        "Data Type": ["Abundance"] * len(receptors_of_interest),
        "KL Divergence": KL_div_vals,
        "EMD": EMD_vals
    }
    metrics_df2_abundance = pd.DataFrame(metrics_abundance)

    # Melt the dataframes for Seaborn plotting
    metrics_df2_raw_melted = metrics_df2_raw.melt(
        id_vars=["Receptor Pair", "Data Type"],
        value_vars=["KL Divergence", "EMD"],
        var_name="Metric",
        value_name="Value"
    )

    metrics_df2_abundance_melted = metrics_df2_abundance.melt(
        id_vars=["Receptor Pair", "Data Type"],
        value_vars=["KL Divergence", "EMD"],
        var_name="Metric",
        value_name="Value"
    )

    # Plotting
    sns.barplot(data=metrics_df2_raw_melted, x='Receptor Pair', y='Value', hue='Metric', ax=ax[0])
    ax[0].set_title('KL Divergence and EMD for 2D Metrics (Raw Data)')
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, ha='right')
    ax[0].set_ylabel('Value')

    sns.barplot(data=metrics_df2_abundance_melted, x='Receptor Pair', y='Value', hue='Metric', ax=ax[1])
    ax[1].set_title('KL Divergence and EMD for 2D Metrics (Receptor Abundance)')
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, ha='right')
    ax[1].set_ylabel('Value')

    '''
    sns.barplot(data=metrics_df1_raw.melt(id_vars=["Receptor Pair", "Data Type"], value_vars=["KL Divergence", "EMD"], var_name="Metric", value_name="Value"), 
                x='Receptor Pair', y='Value', hue='Metric', ax=ax[2])
    ax[2].set_title('KL Divergence and EMD for 1D Metrics (Raw Data)')
    ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation=45, ha='right')
    ax[2].set_ylabel('Value')

    sns.barplot(data=metrics_df1_abundance.melt(id_vars=["Receptor Pair", "Data Type"], value_vars=["KL Divergence", "EMD"], var_name="Metric", value_name="Value"), 
                x='Receptor Pair', y='Value', hue='Metric', ax=ax[3])
    ax[3].set_title('KL Divergence and EMD for 1D Metrics (Receptor Abundance)')
    ax[3].set_xticklabels(ax[3].get_xticklabels(), rotation=45, ha='right')
    ax[3].set_ylabel('Value')'
    '''

    return f