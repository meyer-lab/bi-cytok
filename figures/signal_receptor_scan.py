import os
import numpy as np
import pandas as pd
from bicytok.imports import importCITE
from bicytok.scanning_funcs import scan_selectivity

MIN_AVG_COUNT = 5
DOSE = 1e-10
VALENCY = np.array([[2, 1, 1]])
CELL_CATEGORIZATION = "CellType2"
SAMPLE_SIZE = 1000
RESULTS_DIR = "../bicytok/data/selectivity_scan_results"

def filter_receptors_by_expression(cite_df, min_avg_count):
    """Filter receptors by minimum average expression."""
    exclude_cols = ["Cell", "CellType1", "CellType2", "CellType3"]
    epitopes = cite_df.columns.tolist()
    receptor_candidates = [ep for ep in epitopes if ep not in exclude_cols]
    
    mean_expr = cite_df[receptor_candidates].mean(axis=0)
    selected_receptors = mean_expr[mean_expr >= min_avg_count].index.tolist()
    
    assert len(selected_receptors) > 0, "No receptors pass the expression threshold"
    return selected_receptors

def prepare_data(cite_df, receptors, signal_idx, cell_categorization):
    """Prepare receptor abundance matrix and cell type labels."""
    epitopes_df = cite_df[receptors + [cell_categorization]].copy()
    epitopes_df = epitopes_df.rename(columns={cell_categorization: "Cell Type"})
    
    rec_abundances = epitopes_df.drop(columns=["Cell Type"]).to_numpy()
    cell_type_labels = epitopes_df["Cell Type"].tolist()
    cell_types = list(set(cell_type_labels))
    
    return rec_abundances, cell_type_labels, cell_types

def flatten_scan_results(opt_selec, opt_affs, opt_Kx_star, receptors, cell_types, signal_receptor):
    """Flatten 3D scan results into tabular format."""
    flattened_data = []
    row_indices, col_indices = np.tril_indices(len(receptors), k=0)
    
    for i, cell_type in enumerate(cell_types):
        for rec1_idx, rec2_idx in zip(row_indices, col_indices):
            rec1_name = receptors[rec1_idx]
            rec2_name = receptors[rec2_idx]
            
            selectivity_val = opt_selec[rec1_idx, rec2_idx, i]
            Kx_star_val = opt_Kx_star[rec1_idx, rec2_idx, i]
            
            affinities = [opt_affs[rec1_idx, rec2_idx, i, j] for j in range(opt_affs.shape[-1])]
            
            row_data = {
                'Signal_Receptor': signal_receptor,
                'Cell_Type': cell_type,
                'Receptor_1': rec1_name,
                'Receptor_2': rec2_name,
                'Selectivity': selectivity_val,
                'Kx_star': Kx_star_val,
            }
            
            for j, aff_val in enumerate(affinities):
                row_data[f'Affinity_Receptor_{j}'] = aff_val
            
            flattened_data.append(row_data)
    
    return flattened_data

def run_expanded_scan():
    """Run selectivity scan with each receptor as signal receptor."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    cite_df = importCITE()
    receptors = filter_receptors_by_expression(cite_df, MIN_AVG_COUNT)
    receptors = receptors[:2]

    # cell_types = ["Treg", "dnT", "CD4 CTL", "CD8 Proliferating", "CD4 TCM", "B memory", "CD8 TCM", "CD8 Naive", "CD4 Proliferating", "NK_CD56bright", "CD4 Naive", "CD8 TEM", "ILC", "CD4 TEM", "B naive"]
    cell_types = ["Treg", "dnT", "CD4 Naive"]

    print(f"Total receptors passing threshold: {len(receptors)}")
    print(f"Total receptor pairs: {len(receptors)*(len(receptors)+1)//2}")
    
    all_results = []
    
    for signal_idx, signal_receptor in enumerate(receptors):
        print(f"\nScanning with signal receptor: {signal_receptor} ({signal_idx+1}/{len(receptors)})")
        
        rec_abundances, cell_type_labels, _ = prepare_data(
            cite_df, receptors, signal_idx, CELL_CATEGORIZATION
        )
        
        print(f"  Cell types: {len(cell_types)}")
        
        opt_selec, opt_affs, opt_Kx_star = scan_selectivity(
            rec_abundances,
            cell_type_labels,
            cell_types,
            dim=2,
            dose=DOSE,
            valencies=VALENCY,
            sample_size=SAMPLE_SIZE,
            signal_col=signal_idx,
        )
        
        flattened_data = flatten_scan_results(
            opt_selec, opt_affs, opt_Kx_star, receptors, cell_types, signal_receptor
        )
        all_results.extend(flattened_data)
        
        print(f"  Processed {len(flattened_data)} receptor pair-cell type combinations")
    
    results_df = pd.DataFrame(all_results)
    output_path = f"{RESULTS_DIR}/expanded_selectivity_scan.csv"
    results_df.to_csv(output_path, index=False)
    
    print(f"\nCompleted scan across {len(receptors)} signal receptor designations")
    print(f"Total results: {len(all_results)} combinations")
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    run_expanded_scan()
