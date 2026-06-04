"""
Functions for scanning across receptor-cell type combinations and saving/loading the
results.
"""

import argparse
import os
import yaml
import numpy as np
import pandas as pd

from bicytok.imports import importCITE
from bicytok.scanning_funcs import scan_selectivity, scan_KL_EMD, sample_cells

def run_selectivity_scan():
    """
    Process CITE-seq data and run selectivity scanning functions and saves results.

    Takes input parameter "output-path" from command line.

    Called with "uv run selectivity_scan --output-path <path>".

    Args:
        output_path (str): Path to save the flattened CSV file containing scan results.

    Returns:
        opt_selec (np.ndarray): 3D array of selectivity values indexed by receptor
            pairs and cell types.
        opt_affs (np.ndarray): 4D array of optimized affinities indexed by receptor
            pairs, cell types, and affinity parameters.
        opt_Kx_star (np.ndarray): 3D array of optimized Kx* values indexed by receptor
            pairs and cell types.
        receptors (list): List of receptors included in the scan after filtering.
        cell_types (list): List of cell types included in the scan after filtering.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", default="/home/sama/receptor_sweep_data/tmp_selectivity_scan.csv")
    args = parser.parse_args()
    output_path = args.output_path

    # General parameters (match distribution metric scans)
    cell_categorization = "CellType2" # Cell types column to use
    sample_size = 1000
    min_avg_count = 5 # Expression threshold
    receptors = None # Receptors to analyze; list or None for all

    cell_types = None # Cell types to analyze; list or None for all
    targ_cell_types = None # Target cell types for selectivity calculation; list or None for all
    exclude_cell_types = False # Boolean to exclude cell types not in cell_types list

    # Binding model parameters
    dose = 1e-10
    valency = np.array([[1, 1, 1]])
    init = [6.0, 7.0, 7.0, -9.0] # Initial optimization values
    signal = "prototype" # Define signal receptor; "prototype" or receptor name
    asym_targs = False # Calculates both symmetric cases (rec1, rec2) and (rec2, rec1)

    # Load and define receptor set
    CITE_DF = importCITE()
    epitopes = CITE_DF.columns.tolist()
    exclude_cols = ["Cell", "CellType1", "CellType2", "CellType3"]
    if receptors is None:
        receptors = [ep for ep in epitopes if ep not in exclude_cols]
    assert signal == "prototype" or signal in receptors, f"Signal receptor '{signal}' not found in receptors list"
    
    # Filter lowly expressed receptors
    mean_expr = CITE_DF[receptors].mean(axis=0)
    selected_receptors = mean_expr[mean_expr >= min_avg_count].index.tolist()
    assert len(selected_receptors) > 0, "No receptors pass the expression threshold"
    assert signal == "prototype" or signal in selected_receptors, f"Signal receptor '{signal}' not found in selected receptors after filtering"
    epitopes_df = CITE_DF[selected_receptors + [cell_categorization]]
    epitopes_df = epitopes_df.rename(columns={cell_categorization: "Cell Type"})
    receptors = selected_receptors

    # Define signal receptor for binding model selectivity ratio
    if signal == "prototype":
        # Insert prototype signal receptor with normally distributed expression values
        rng = np.random.default_rng()
        prototype_signal_receptor = rng.normal(loc=50, scale=5, size=(epitopes_df.shape[0],))
        prototype_signal_receptor = np.clip(prototype_signal_receptor, a_min=0, a_max=None)
        epitopes_df.insert(0, "Prototype_Signal_Receptor", prototype_signal_receptor)
        receptors = ["Prototype_Signal_Receptor"] + receptors
        signal_ind = 0
    else:
        signal_ind = receptors.index(signal)
    
    # Filter out unused cell types
    if cell_types is not None and exclude_cell_types:
        epitopes_df = epitopes_df[epitopes_df["Cell Type"].isin(cell_types)]

    rec_abundances = epitopes_df.drop(columns=["Cell Type"]).to_numpy()
    cell_type_labels = epitopes_df["Cell Type"].tolist()
    if cell_types is None:
        cell_types = list(set(cell_type_labels))

    if targ_cell_types is None:
        targ_cell_types = cell_types
    
    print(f"Using receptors: {receptors}, total {len(receptors)} receptors.")
    print(f"Total possible receptor pairs: {len(receptors) * (len(receptors) + 1) // 2}.")
    print(f"Off-target cell types considered: {cell_types}, total {len(cell_types)} cell types.")
    print(f"Target cell types for selectivity: {targ_cell_types}, total {len(targ_cell_types)} cell types.")
    
    opt_selec, opt_affs, opt_Kx_star = scan_selectivity(
        rec_abundances,
        cell_type_labels,
        targ_cell_types,
        dim=2,
        dose=dose,
        valencies=valency,
        sample_size=sample_size,
        signal_col=signal_ind,
        init_method=init,
        asym_targs=asym_targs,
    )

    # Save flattened results
    flattened_data = []
    for i, cell_type in enumerate(targ_cell_types):
        for rec1_idx, rec2_idx in np.ndindex(len(receptors), len(receptors)):
            rec1_name = receptors[rec1_idx]
            rec2_name = receptors[rec2_idx]
            selectivity_val = opt_selec[rec1_idx, rec2_idx, i]
            Kx_star_val = opt_Kx_star[rec1_idx, rec2_idx, i]
            affinities = [
                opt_affs[rec1_idx, rec2_idx, i, j] for j in range(opt_affs.shape[-1])
            ]
            row_data = {
                "Cell_Type": cell_type,
                "Receptor_1": rec1_name,
                "Receptor_2": rec2_name,
                "Selectivity": selectivity_val,
                "Kx_star": Kx_star_val,
            }
            for j, aff_val in enumerate(affinities):
                row_data[f"Affinity_Receptor_{j}"] = aff_val
            flattened_data.append(row_data)
    flattened_df = pd.DataFrame(flattened_data)
    flattened_df.to_csv(output_path, index=False)

    yaml_path = os.path.splitext(output_path)[0] + ".yaml"
    scan_params = {
        "scan_type": "selectivity",
        "output_csv": output_path,
        "general": {
            "cell_categorization": cell_categorization,
            "sample_size": sample_size,
            "min_expression_threshold": min_avg_count,
            "exclude_unused_cell_types": exclude_cell_types,
            "dim": 2,
        },
        "binding_model": {
            "dose": float(dose),
            "complex_valency_signal-target1-target2": valency.tolist(),
            "initial_affinities": init,
            "signal_receptor": signal,
            "asym_targs": asym_targs,
        },
        "receptors_used_before_filtering": receptors,
        "n_receptors": len(receptors),
        "n_receptor_pairs": len(receptors) * (len(receptors) + 1) // 2,
        "off_target_cell_types": cell_types,
        "target_cell_types": targ_cell_types,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(scan_params, f, default_flow_style=False, sort_keys=False)

    return opt_selec, opt_affs, opt_Kx_star, receptors, cell_types


def run_KL_EMD_scan():
    """
    Process CITE-seq data and run KL and EMD scanning functions and saves results.

    Takes input parameter "output-path" from command line.

    Called with "uv run KL_EMD_scan --output-path <path>".

    Args:
        output_path (str): Path to save the flattened CSV file containing scan results.

    Returns:
        KL_results (np.ndarray): 3D array of KL divergence values indexed by receptor
            pairs and cell types.
        EMD_results (np.ndarray): 3D array of EMD values indexed by receptor
            pairs and cell types.
        receptors (list): List of receptors included in the scan after filtering.
        cell_types (list): List of cell types included in the scan after filtering.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", default="/home/sama/receptor_sweep_data/tmp_KL-EMD_scan.csv")
    args = parser.parse_args()
    output_path = args.output_path
    
    # General parameters (match selectivity scans)
    cell_categorization = "CellType2" # Cell types column to use
    sample_size = 1000
    min_avg_count = 5 # Expression threshold
    receptors = None # Receptors to analyze; list or None for all
    cell_types = None # Cell types to analyze; list or None for all
    targ_cell_types = None # Target cell types for selectivity calculation; list or None for all
    exclude_cell_types = False # Boolean to exclude cell types not in cell_types list

    # Distance metric scan parameters
    filter_by_target_expr = False # Boolean to filter out receptors with higher off-target expression

    # Load and define receptor set
    CITE_DF = importCITE()
    epitopes = CITE_DF.columns.tolist()
    exclude_cols = ["Cell", "CellType1", "CellType2", "CellType3"]
    if receptors is None:
        receptors = [ep for ep in epitopes if ep not in exclude_cols]
    
    # Filter lowly expressed receptors
    mean_expr = CITE_DF[receptors].mean(axis=0)
    selected_receptors = mean_expr[mean_expr >= min_avg_count].index.tolist()
    assert len(selected_receptors) > 0, "No receptors pass the expression threshold"
    epitopes_df = CITE_DF[selected_receptors + [cell_categorization]]
    epitopes_df = epitopes_df.rename(columns={cell_categorization: "Cell Type"})
    receptors = selected_receptors

    # Filter out unused cell types
    if cell_types is not None and exclude_cell_types:
        epitopes_df = epitopes_df[epitopes_df["Cell Type"].isin(cell_types)]

    rec_abundances = epitopes_df.drop(columns=["Cell Type"]).to_numpy()
    cell_type_labels = epitopes_df["Cell Type"].tolist()
    if cell_types is None:
        cell_types = list(set(cell_type_labels))

    if targ_cell_types is None:
        targ_cell_types = cell_types
    
    print(f"Using receptors: {receptors}, total {len(receptors)} receptors.")
    print(f"Total possible receptor pairs: {len(receptors) * (len(receptors) + 1) // 2}.")
    print(f"Off-target cell types considered: {cell_types}, total {len(cell_types)} cell types.")
    print(f"Target cell types for selectivity: {targ_cell_types}, total {len(targ_cell_types)} cell types.")

    KL_results, EMD_results = scan_KL_EMD(
        rec_abundances,
        cell_type_labels,
        targ_cell_types,
        dim=2,
        sample_size=sample_size,
        filter_by_target_expr=filter_by_target_expr,
    )

    # Save flattened results
    flattened_data = []
    for i, cell_type in enumerate(targ_cell_types):
        for rec1_idx, rec2_idx in np.ndindex(len(receptors), len(receptors)):
            rec1_name = receptors[rec1_idx]
            rec2_name = receptors[rec2_idx]
            KL_val = KL_results[rec1_idx, rec2_idx, i]
            EMD_val = EMD_results[rec1_idx, rec2_idx, i]
            flattened_data.append({
                "Cell_Type": cell_type,
                "Receptor_1": rec1_name,
                "Receptor_2": rec2_name,
                "KL_Divergence": KL_val,
                "EMD": EMD_val,
            })
    flattened_df = pd.DataFrame(flattened_data)
    flattened_df.to_csv(output_path, index=False)

    yaml_path = os.path.splitext(output_path)[0] + ".yaml"
    scan_params = {
        "scan_type": "KL_EMD",
        "output_csv": output_path,
        "general": {
            "cell_categorization": cell_categorization,
            "sample_size": sample_size,
            "min_expression_threshold": min_avg_count,
            "exclude_unused_cell_types": exclude_cell_types,
            "dim": 2,
        },
        "distance_metric": {
            "filter_by_target_expr": filter_by_target_expr,
        },
        "receptors_used_before_filtering": receptors,
        "n_receptors": len(receptors),
        "n_receptor_pairs": len(receptors) * (len(receptors) + 1) // 2,
        "off_target_cell_types": cell_types,
        "target_cell_types": targ_cell_types,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(scan_params, f, default_flow_style=False, sort_keys=False)

    return KL_results, EMD_results, receptors, cell_types


def load_selec_scan_results(results_path):
    """
    Load flattened binding model results from CSV file.

    Args:
        results_path (str): Path to the flattened CSV file containing scan results.
    
    Returns:
        opt_selec (np.ndarray): 3D array of selectivity values indexed by receptor
            pairs and cell types.
        opt_affs (np.ndarray): 4D array of optimized affinities indexed by receptor
            pairs, cell types, and affinity parameters.
        opt_Kx_star (np.ndarray): 3D array of optimized Kx* values indexed by receptor
            pairs and cell types.
        all_receptors (list): List of all receptors included in the scan.
        cell_types_loaded (list): List of all cell types included in the scan.
    """

    scan_data = pd.read_csv(results_path)

    # Extract unique receptors and cell types
    all_receptors = sorted(
        set(scan_data["Receptor_1"].unique()) | set(scan_data["Receptor_2"].unique())
    )
    cell_types_loaded = sorted(scan_data["Cell_Type"].unique())

    # Create receptor index mapping
    receptor_to_idx = {rec: idx for idx, rec in enumerate(all_receptors)}
    
    # Reconstruct opt_selec matrix
    n_affinities = len([col for col in scan_data.columns if col.startswith("Affinity_Receptor_")])
    opt_selec = np.full(
        (len(all_receptors), len(all_receptors), len(cell_types_loaded)), np.nan
    )
    opt_affs = np.full(
        (len(all_receptors), len(all_receptors), len(cell_types_loaded), n_affinities), np.nan
    )
    opt_Kx_star = np.full(
        (len(all_receptors), len(all_receptors), len(cell_types_loaded)), np.nan
    )
    for _, row in scan_data.iterrows():
        rec1_idx = receptor_to_idx[row["Receptor_1"]]
        rec2_idx = receptor_to_idx[row["Receptor_2"]]
        cell_type_idx = cell_types_loaded.index(row["Cell_Type"])
        opt_selec[rec1_idx, rec2_idx, cell_type_idx] = row["Selectivity"]
        opt_Kx_star[rec1_idx, rec2_idx, cell_type_idx] = row["Kx_star"]
        for j in range(3):
            aff_col = f"Affinity_Receptor_{j}"
            if aff_col in row:
                opt_affs[rec1_idx, rec2_idx, cell_type_idx, j] = row[aff_col]

    return opt_selec, opt_affs, opt_Kx_star, all_receptors, cell_types_loaded


def load_KL_EMD_scan_results(results_path):
    """
    Load flattened KL and EMD scan results from CSV file.

    Args:
        results_path (str): Path to the flattened CSV file containing scan results.
    
    Returns:
        KL_results (np.ndarray): 3D array of KL divergence values indexed by receptor
            pairs and cell types.
        EMD_results (np.ndarray): 3D array of EMD values indexed by receptor
            pairs and cell types.
        all_receptors (list): List of all receptors included in the scan.
        cell_types_loaded (list): List of all cell types included in the scan.
    """

    scan_data = pd.read_csv(results_path)

    # Extract unique receptors and cell types
    all_receptors = sorted(
        set(scan_data["Receptor_1"].unique()) | set(scan_data["Receptor_2"].unique())
    )
    cell_types_loaded = sorted(scan_data["Cell_Type"].unique())

    # Create receptor index mapping
    receptor_to_idx = {rec: idx for idx, rec in enumerate(all_receptors)}

    # Reconstruct KL and EMD matrices
    KL_results = np.full(
        (len(all_receptors), len(all_receptors), len(cell_types_loaded)), np.nan
    )
    EMD_results = np.full(
        (len(all_receptors), len(all_receptors), len(cell_types_loaded)), np.nan
    )
    for _, row in scan_data.iterrows():
        rec1_idx = receptor_to_idx[row["Receptor_1"]]
        rec2_idx = receptor_to_idx[row["Receptor_2"]]
        cell_type_idx = cell_types_loaded.index(row["Cell_Type"])
        KL_results[rec1_idx, rec2_idx, cell_type_idx] = row["KL_Divergence"]
        EMD_results[rec1_idx, rec2_idx, cell_type_idx] = row["EMD"]

    return KL_results, EMD_results, all_receptors, cell_types_loaded


def filter_scan_by_target_expr():
    """
    Post-hoc filter scan results to remove receptor pairs where either receptor has
    higher mean expression in off-target cells than in target cells.

    Replicates the filter_by_target_expr step using the sampling parameters stored in
    the corresponding YAML file. Metric values for filtered-out pairs are set to NaN.
    Receptors not present in the CITE data (e.g., Prototype_Signal_Receptor) are
    always treated as valid.

    Called with "uv run filt_scan --results-path <path> --filter-id <filter_id>".

    Args:
        results_path (str): Path to the scan CSV file. Must have matching YAML file
            with scan parameters in the same directory.
            E.g., "/home/sama/receptor_sweep_data/tmp_selec_scan.csv"
        filter_id (str): Suffix appended before the file extension in the output path.

    Returns:
        str: Path to the saved filtered CSV file.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path")
    parser.add_argument("--filter-id", default="filt")
    args = parser.parse_args()
    results_path = args.results_path
    filter_id = args.filter_id

    yaml_path = os.path.splitext(results_path)[0] + ".yaml"
    assert os.path.exists(yaml_path), f"YAML file with scan parameters not found at {yaml_path}"
    with open(yaml_path) as f:
        scan_params = yaml.safe_load(f)

    scan_type = scan_params["scan_type"]
    cell_categorization = scan_params["general"]["cell_categorization"]
    sample_size = scan_params["general"]["sample_size"]
    targ_cell_types = scan_params["target_cell_types"]
    min_avg_count = scan_params["general"]["min_expression_threshold"]

    scan_data = pd.read_csv(results_path)

    if scan_type == "selectivity":
        metric_cols = ["Selectivity", "Kx_star"] + [
            c for c in scan_data.columns if c.startswith("Affinity_Receptor_")
        ]
        balance = False
    elif scan_type == "KL_EMD":
        metric_cols = ["KL_Divergence", "EMD"]
        balance = True
    else:
        raise ValueError(f"Unknown scan_type '{scan_type}' in {yaml_path}")

    CITE_DF = importCITE()
    exclude_cols = ["Cell", "CellType1", "CellType2", "CellType3"]
    cite_receptors = [ep for ep in CITE_DF.columns if ep not in exclude_cols]
    mean_expr = CITE_DF[cite_receptors].mean(axis=0)
    cite_receptors = mean_expr[mean_expr >= min_avg_count].index.tolist()

    epitopes_df = CITE_DF[cite_receptors + [cell_categorization]].rename(
        columns={cell_categorization: "Cell Type"}
    )
    rec_abundances = epitopes_df.drop(columns=["Cell Type"]).to_numpy()
    cell_type_labels = epitopes_df["Cell Type"].tolist()

    # Receptors absent from CITE_DF (e.g. Prototype_Signal_Receptor) are always valid
    all_scan_receptors = (
        set(scan_data["Receptor_1"].unique()) | set(scan_data["Receptor_2"].unique())
    )
    non_cite_receptors = all_scan_receptors - set(cite_receptors)

    for cell_type in targ_cell_types:
        sampled_rec_abundances, sampled_cell_type_labels = sample_cells(
            rec_abundances, cell_type_labels, cell_type,
            sample_size=sample_size, balance=balance,
        )

        targ_mask = sampled_cell_type_labels == cell_type
        off_targ_mask = ~targ_mask
        mean_targ = sampled_rec_abundances[targ_mask, :].mean(axis=0)
        mean_off_targ = sampled_rec_abundances[off_targ_mask, :].mean(axis=0)

        valid_receptors = (
            {rec for rec, mt, mo in zip(cite_receptors, mean_targ, mean_off_targ) if mt > mo}
            | non_cite_receptors
        )

        ct_mask = scan_data["Cell_Type"] == cell_type
        invalid_mask = ct_mask & (
            ~scan_data["Receptor_1"].isin(valid_receptors)
            | ~scan_data["Receptor_2"].isin(valid_receptors)
        )
        scan_data.loc[invalid_mask, metric_cols] = np.nan
        print(
            f"Filtered {invalid_mask.sum()} / {ct_mask.sum()} pairs for {cell_type} "
            f"({len(valid_receptors)} / {len(cite_receptors) + len(non_cite_receptors)} valid receptors)."
        )

    base, ext = os.path.splitext(results_path)
    filtered_path = f"{base}_{filter_id}{ext}"
    scan_data.to_csv(filtered_path, index=False)
    print(f"Saved filtered scan to {filtered_path}")

    return filtered_path
