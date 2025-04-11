import seaborn as sns
from pathlib import Path
import numpy as np
import pandas as pd
from ..distance_metric_funcs import KL_EMD_2D, KL_EMD_1D
from ..imports import importCITE, sample_receptor_abundances, calc_conv_facts
from .common import getSetup
from ..selectivity_funcs import optimize_affs


path_here = Path(__file__).parent.parent

def makeFigure():
    ax, f = getSetup((5, 5), (1, 1))

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
    ]
    sample_size = 500
    cell_categorization = "CellType2"
    test_valencies = [(1), (2)]
    dose = 10e-2

    CITE_DF = importCITE()

    epitopes = [
        col
        for col in CITE_DF.columns
        if col not in ["CellType1", "CellType2", "CellType3"]
    ]
    epitopesDF = CITE_DF[epitopes + [cell_categorization]]
    epitopesDF = epitopesDF.rename(columns={cell_categorization: "Cell Type"})

    ##
    
    target_cells = epitopesDF[epitopesDF["Cell Type"] == targCell]
    off_target_cells = epitopesDF[epitopesDF["Cell Type"] != targCell]
    num_target_cells = sample_size // 2
    num_off_target_cells = sample_size - num_target_cells

    sampled_target_cells = target_cells.sample(
        min(num_target_cells, target_cells.shape[0]), random_state=42
    )
    sampled_off_target_cells = off_target_cells.sample(
        min(num_off_target_cells, off_target_cells.shape[0]),
        random_state=42,
    )

    sampleDF = pd.concat([sampled_target_cells, sampled_off_target_cells])
    raw_sampleDF = sampleDF

    # Calculate conversion factors for each epitope
    convFactDict, defaultConvFact = calc_conv_facts()

    convFacts = [convFactDict.get(epitope, defaultConvFact) for epitope in epitopes]

    sampleDF[epitopes] = sampleDF[epitopes] * convFacts

    filtered_sampleDF = sampleDF.loc[
        :,
        sampleDF.columns.str.fullmatch("|".join(receptors_of_interest), case=False),
    ]
    receptors_of_interest = filtered_sampleDF.columns

    on_target_mask = (sampleDF["Cell Type"] == targCell).to_numpy()
    off_target_mask = ~on_target_mask

    rec_abundances = filtered_sampleDF.to_numpy()
    
    target_cells = epitopesDF[epitopesDF["Cell Type"] == targCell]
    num_target_cells = min(sample_size, target_cells.shape[0])
    
    filtered_sampleDFraw = raw_sampleDF.loc[
        :,
        raw_sampleDF.columns.str.fullmatch("|".join(receptors_of_interest), case=False),
    ]
    
    rec_abundancesraw = filtered_sampleDFraw[receptors_of_interest].to_numpy()
    avg_selectivity_raw = []
    avg_selectivity_converted = []
    receptor_labels = []

    for i, signal_receptor in enumerate(receptors_of_interest):
        selectivity_vals_converted = []
        selectivity_vals_raw = []

        for receptor in receptors_of_interest:
            if receptor == signal_receptor:
                continue

            for valency in test_valencies:
                model_valencies = np.array([[valency, valency]])

                # Converted
                targRecs = sampleDF[[signal_receptor, receptor]].to_numpy()
                offTargRecs = sampleDF[[signal_receptor, receptor]].to_numpy()
                optSelec_converted, _ = optimize_affs(
                    targRecs=targRecs[on_target_mask],
                    offTargRecs=offTargRecs[off_target_mask],
                    dose=dose,
                    valencies=model_valencies,
                )
                selectivity_vals_converted.append(1 / optSelec_converted)

                # Raw
                targRecs_raw = raw_sampleDF[[signal_receptor, receptor]].to_numpy()
                offTargRecs_raw = raw_sampleDF[[signal_receptor, receptor]].to_numpy()
                optSelec_raw, _ = optimize_affs(
                    targRecs=targRecs_raw[on_target_mask],
                    offTargRecs=offTargRecs_raw[off_target_mask],
                    dose=dose,
                    valencies=model_valencies,
                )
                selectivity_vals_raw.append(1 / optSelec_raw)

        avg_selectivity_converted.append(np.mean(selectivity_vals_converted))
        avg_selectivity_raw.append(np.mean(selectivity_vals_raw))
        receptor_labels.append(signal_receptor)

    x = np.arange(len(receptor_labels))  
    width = 0.35  

    ax[0].bar(x - width/2, avg_selectivity_raw, width, label='Raw Selectivity')
    ax[0].bar(x + width/2, avg_selectivity_converted, width, label='Converted Selectivity')

    ax[0].set_ylabel('Average 1 / Selectivity')
    ax[0].set_title('Raw vs Converted Selectivity per Signal Receptor')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(receptor_labels, rotation=45)
    ax[0].legend()
    
    '''
    
    KL_div_vals, EMD_vals = KL_EMD_2D(
        rec_abundances, on_target_mask, off_target_mask, calc_1D=True
    )
    KL_div_valsRAW, EMD_valsRAW = KL_EMD_2D(
        rec_abundancesraw, on_target_mask, off_target_mask, calc_1D=True
    )
    KL_div_valsRAW1, EMD_valsRAW1 = KL_EMD_1D(
        rec_abundancesraw, on_target_mask, off_target_mask
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
    '''

    return f 