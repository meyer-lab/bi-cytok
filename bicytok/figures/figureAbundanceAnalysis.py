import seaborn as sns
from pathlib import Path
import numpy as np
import pandas as pd
from ..distance_metric_funcs import KL_EMD_2D, KL_EMD_1D
from ..imports import importCITE, sample_receptor_abundances
from .common import getSetup

path_here = Path(__file__).parent.parent

def makeFigure():
    ax, f = getSetup((15, 5), (1, 3))

    # Parameters
    targCell = "Treg"
    receptors_of_interest = [
        "CD25",
        "CD4-1",
        "CD27",
        "CD4-2",
        "CD278",
        "CD122",
        "CD28",
        "TCR-2",
        "TIGIT",
        "TSLPR",
    ]
    sample_size = 1000
    cell_categorization = "CellType2"

    CITE_DF = importCITE()

    assert targCell in CITE_DF[cell_categorization].unique()

    epitopesDF = CITE_DF[receptors_of_interest + [cell_categorization]]
    epitopesDF = epitopesDF.rename(columns={cell_categorization: "Cell Type"})
    sampleDF = sample_receptor_abundances(
        CITE_DF=epitopesDF,
        numCells=sample_size,
        targCellType=targCell,
    )
    rec_abundances = sampleDF[receptors_of_interest].to_numpy()

    target_mask = (sampleDF["Cell Type"] == targCell).to_numpy()
    off_target_mask = ~target_mask

    target_cells = epitopesDF[epitopesDF["Cell Type"] == targCell]
    num_target_cells = min(sample_size, target_cells.shape[0])
    raw_sampleDF = target_cells.sample(num_target_cells, random_state=42)
    rec_abundancesraw = raw_sampleDF[receptors_of_interest].to_numpy()

    KL_div_vals, EMD_vals = KL_EMD_2D(
        rec_abundances, target_mask, off_target_mask, calc_1D=True
    )
    KL_div_valsRAW, EMD_valsRAW = KL_EMD_2D(
        rec_abundancesraw, target_mask, off_target_mask, calc_1D=True
    )
    KL_div_valsRAW1, EMD_valsRAW1 = KL_EMD_1D(
        rec_abundancesraw, target_mask, off_target_mask
    )
    
    KL_div_vals_2Draw = np.diag(KL_div_valsRAW)
    EMD_vals_2Draw = np.diag(EMD_valsRAW)


    KL_div_vals_2D = np.diag(KL_div_vals)
    EMD_vals_2D = np.diag(EMD_vals
    )
    data = {
        "Receptor": receptors_of_interest,
        "KL Divergence RAW 2D": KL_div_vals_2Draw,
        "EMD RAW 2D": EMD_vals_2Draw,
        "KL Divergence": KL_div_vals_2D,
        "EMD": EMD_vals_2D,
        "KL Divergence RAW 1D": KL_div_valsRAW1,
        "EMD RAW 1D": EMD_valsRAW1,
    }
    df = pd.DataFrame(data)
    df_melted_raw = df[["Receptor", "KL Divergence RAW 2D", "EMD RAW 2D"]].melt(id_vars="Receptor", 
                                                                         value_vars=["KL Divergence RAW 2D", "EMD RAW 2D"], 
                                                                         var_name="Metric", 
                                                                         value_name="Value")

    df_melted_abundance = df[["Receptor", "KL Divergence", "EMD"]].melt(id_vars="Receptor", 
                                                                        value_vars=["KL Divergence", "EMD"], 
                                                                        var_name="Metric", 
                                                                        value_name="Value")
    
    df_melted_raw1d = df[["Receptor", "KL Divergence RAW 1D", "EMD RAW 1D"]].melt(id_vars="Receptor", 
                                                                         value_vars=["KL Divergence RAW 1D", "EMD RAW 1D"], 
                                                                         var_name="Metric", 
                                                                         value_name="Value")

    # Plot barplot for RAW data
    sns.barplot(data=df_melted_raw, x="Receptor", y="Value", hue="Metric", ax=ax[0])
    ax[0].set_title(f"Receptor KL Divergence and EMD (RAW) 2D")
    ax[0].set_xlabel("Receptor")
    ax[0].set_ylabel("Value")
    ax[0].legend(title="Metric")

    # Plot barplot for Abundance conversion data
    sns.barplot(data=df_melted_abundance, x="Receptor", y="Value", hue="Metric", ax=ax[1])
    ax[1].set_title(f"Receptor KL Divergence and EMD (Abundance Conversion) 2D")
    ax[1].set_xlabel("Receptor")
    ax[1].set_ylabel("Value")
    ax[1].legend(title="Metric")

    sns.barplot(data=df_melted_raw1d, x="Receptor", y="Value", hue="Metric", ax=ax[2])
    ax[2].set_title(f"Receptor KL Divergence and EMD (RAW) 1D")
    ax[2].set_xlabel("Receptor")
    ax[2].set_ylabel("Value")
    ax[2].legend(title="Metric")

    return f 