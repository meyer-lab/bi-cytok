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
    
    sampleDFrawvalues = epitopesDF.sample(
        min(sample_size, epitopesDF.shape[0]), random_state=42
    )

    # Ensure the final DataFrame has the same format as `sampleDF`, including the 'Cell Type' column
    sampleDFrawvalues = sampleDFrawvalues.rename(columns={cell_categorization: "Cell Type"})

    
    filtered_sampleDF = sampleDF.loc[
        :,
        sampleDF.columns.str.fullmatch("|".join(receptors_of_interest), case=False),
    ]
    receptors_of_interest = filtered_sampleDF.columns
    filtered_sampleDFrawvalues = sampleDFrawvalues.loc[
        :,
        sampleDFrawvalues.columns.str.fullmatch("|".join(receptors_of_interest), case=False),
    ]

    on_target_mask = (sampleDF["Cell Type"] == targCell).to_numpy()
    off_target_mask = ~on_target_mask

    rec_abundances = filtered_sampleDF.to_numpy()
    rec_abdundnaces_raw = filtered_sampleDFrawvalues.to_numpy()

    KL_div_vals, EMD_vals = KL_EMD_2D(
        rec_abundances, on_target_mask, off_target_mask, calc_1D=False
    )

    EMD_matrix = np.tril(EMD_vals, k=0)
    EMD_matrix = EMD_matrix + EMD_matrix.T - np.diag(np.diag(EMD_matrix))
    KL_matrix = np.tril(KL_div_vals, k=0)
    KL_matrix = KL_matrix + KL_matrix.T - np.diag(np.diag(KL_matrix))

    KL_div_valsRAW, EMD_valsRAW = KL_EMD_2D(
        rec_abdundnaces_raw, on_target_mask, off_target_mask, calc_1D=False
    )

    EMD_matrixraw = np.tril(EMD_valsRAW, k=0)
    EMD_matrixraw = EMD_matrixraw + EMD_matrixraw.T - np.diag(np.diag(EMD_matrixraw))
    KL_matrixraw = np.tril(KL_div_valsRAW, k=0)
    KL_matrixraw = KL_matrixraw + KL_matrixraw.T - np.diag(np.diag(KL_matrixraw))
    
    '''
    metrics_df2_raw = pd.DataFrame(
        {
            "Receptor Pair": 
            "KL Divergence": 
            "EMD": 
            "Data Type": "Raw Data"
        }
    )

    metrics_df2_abundance = pd.DataFrame(
        {
            "Receptor Pair": 
            "KL Divergence": 
            "EMD": 
            "Data Type": "Receptor Abundance"
        }
    )
    
    metrics_df1_raw = pd.DataFrame(
        {
            "Receptor Pair": 
            "KL Divergence": 
            "EMD": 
            "Data Type": "Raw Data"
        }
    )
    
    metrics_df1_abundance = pd.DataFrame(
        {
            "Receptor Pair": 
            "KL Divergence": 
            "EMD": 
            "Data Type": "Receptor Abundance"
        }
    )
    '''
    print ("metrics_df2_raw", metrics_df2_raw)
    print ("metrics_df2_abundance", metrics_df2_abundance)
    #print ("metrics_df1_raw", metrics_df1_raw)
    #print ("metrics_df1_raw", metrics_df1_raw)
    sns.barplot(data=metrics_df2_raw.melt(id_vars=["Receptor Pair", "Data Type"], value_vars=["KL Divergence", "EMD"], var_name="Metric", value_name="Value"), 
                x='Receptor Pair', y='Value', hue='Metric', ax=ax[0])
    ax[0].set_title('KL Divergence and EMD for 2D Metrics (Raw Data)')
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, ha='right')
    ax[0].set_ylabel('Value')

    sns.barplot(data=metrics_df2_abundance.melt(id_vars=["Receptor Pair", "Data Type"], value_vars=["KL Divergence", "EMD"], var_name="Metric", value_name="Value"), 
                x='Receptor Pair', y='Value', hue='Metric', ax=ax[1])
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