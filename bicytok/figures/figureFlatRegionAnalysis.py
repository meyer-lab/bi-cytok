"""
Analyzes the "flat" regions of selectivity-scaling factor space for all epitopes
to identify optimal universal scaling factors.

This figure extends the BindingModelScaling analysis to:
1. Run selectivity optimization across scaling factors for all available epitopes
2. Identify flat/plateau regions in the selectivity curves using gradient analysis
3. Visualize the range and flatness characteristics of each epitope's plateau
4. Suggest universal scaling factors that work across multiple epitopes

Data Import:
- The CITE-seq dataframe (`importCITE`)

Parameters:
- targCell: cell type whose selectivity will be maximized
- sample_size: number of cells to sample from the CITE-seq dataframe
- cell_categorization: column name in the CITE-seq dataframe that categorizes cells
- model_valencies: valencies each receptor's ligand in the model molecule
- dose: dose of the model molecule
- signal_receptor: receptor that is the target of the model molecule
- affinity_bounds: optimization bounds for the affinities of the receptors
- num_conv_factors: number of conversion factors to test
- flatness_threshold: maximum gradient magnitude to consider "flat"
- min_flat_length: minimum number of consecutive points to consider a flat region

Outputs:
- A multi-panel figure showing:
  1. Selectivity curves for all epitopes with flat regions highlighted
  2. Flat region characteristics (start, end, average selectivity)
  3. Gradient magnitude analysis
  4. Universal scaling factor recommendations
"""

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

from ..imports import importCITE, sample_receptor_abundances, filter_receptor_abundances
from ..selectivity_funcs import optimize_affs
from .common import getSetup

path_here = Path(__file__).parent.parent


def detect_flat_regions(
    x_values: np.ndarray, 
    y_values: np.ndarray, 
    flatness_threshold: float = 0.1,
    min_flat_length: int = 3,
    smooth_sigma: float = 1.0
) -> list[dict]:
    """
    Detect flat regions in a curve using gradient analysis.
    
    Args:
        x_values: x-axis values (scaling factors)
        y_values: y-axis values (selectivity)
        flatness_threshold: maximum gradient magnitude to consider "flat"
        min_flat_length: minimum number of consecutive points for a flat region
        smooth_sigma: gaussian smoothing parameter for gradient calculation
        
    Returns:
        List of dictionaries containing flat region information
    """
    # Smooth the data to reduce noise in gradient calculation
    smoothed_y = gaussian_filter1d(y_values, sigma=smooth_sigma)
    
    # Calculate gradient magnitude
    gradients = np.abs(np.gradient(smoothed_y, x_values))
    
    # Identify flat points
    flat_mask = gradients < flatness_threshold
    
    # Find consecutive flat regions
    flat_regions = []
    in_flat_region = False
    region_start = None
    
    for i, is_flat in enumerate(flat_mask):
        if is_flat and not in_flat_region:
            # Start of new flat region
            region_start = i
            in_flat_region = True
        elif not is_flat and in_flat_region:
            # End of flat region
            region_length = i - region_start
            if region_length >= min_flat_length:
                flat_regions.append({
                    'start_idx': region_start,
                    'end_idx': i - 1,
                    'start_x': x_values[region_start],
                    'end_x': x_values[i - 1],
                    'length': region_length,
                    'avg_selectivity': np.mean(y_values[region_start:i]),
                    'std_selectivity': np.std(y_values[region_start:i]),
                    'avg_gradient': np.mean(gradients[region_start:i])
                })
            in_flat_region = False
    
    # Handle case where curve ends in flat region
    if in_flat_region:
        region_length = len(flat_mask) - region_start
        if region_length >= min_flat_length:
            flat_regions.append({
                'start_idx': region_start,
                'end_idx': len(flat_mask) - 1,
                'start_x': x_values[region_start],
                'end_x': x_values[-1],
                'length': region_length,
                'avg_selectivity': np.mean(y_values[region_start:]),
                'std_selectivity': np.std(y_values[region_start:]),
                'avg_gradient': np.mean(gradients[region_start:])
            })
    
    return flat_regions


def find_universal_scaling_factors(
    epitope_flat_regions: dict, 
    conversion_factors: np.ndarray,
    min_epitopes: int = 2
) -> list[dict]:
    """
    Find scaling factors that put multiple epitopes in their flat regions.
    
    Args:
        epitope_flat_regions: Dictionary mapping epitope names to their flat regions
        conversion_factors: Array of tested conversion factors
        min_epitopes: Minimum number of epitopes that must be in flat region
        
    Returns:
        List of universal scaling factor recommendations
    """
    universal_factors = []
    
    for factor in conversion_factors:
        epitopes_in_flat = []
        
        for epitope, regions in epitope_flat_regions.items():
            for region in regions:
                if region['start_x'] <= factor <= region['end_x']:
                    epitopes_in_flat.append({
                        'epitope': epitope,
                        'selectivity': region['avg_selectivity'],
                        'region': region
                    })
                    break
        
        if len(epitopes_in_flat) >= min_epitopes:
            universal_factors.append({
                'scaling_factor': factor,
                'num_epitopes': len(epitopes_in_flat),
                'epitopes': epitopes_in_flat,
                'avg_selectivity': np.mean([e['selectivity'] for e in epitopes_in_flat]),
                'min_selectivity': np.min([e['selectivity'] for e in epitopes_in_flat])
            })
    
    return universal_factors


def makeFigure():
    ax, f = getSetup((16, 12), (2, 2))

    # Parameters
    targCell = "Treg"
    sample_size = 200
    cell_categorization = "CellType2"
    model_valencies = np.array([[(2), (2)]])
    dose = 10e-2
    signal_receptor = "CD122"
    affinity_bounds = (-10, 25)
    num_conv_factors = 20
    conversion_factors = np.logspace(-2, 8, num=num_conv_factors)
    
    # Flatness detection parameters
    flatness_threshold = 1e-8
    min_flat_length = 3
    
    # Import and prepare data
    CITE_DF = importCITE()
    epitopes_all = [
        col for col in CITE_DF.columns
        if col not in ["CellType1", "CellType2", "CellType3"]
    ]

    epitopes_all = [
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
        "CD146",
        "CD3-1",
        "CD3-2",
        "CD109",
    ]

    epitopesDF = CITE_DF[epitopes_all + [cell_categorization]]
    epitopesDF = epitopesDF.rename(columns={cell_categorization: "Cell Type"})
    
    sampleDF = sample_receptor_abundances(
        CITE_DF=epitopesDF,
        numCells=sample_size,
        targCellType=targCell,
        convert=False,
    )
    
    # Get all available epitopes except signal receptor
    # filtered_sampleDF = filter_receptor_abundances(
    #     sampleDF, targ_cell_type=targCell
    # )
    filtered_sampleDF = pd.DataFrame(
        {
            signal_receptor: sampleDF[signal_receptor],
            "Cell Type": sampleDF["Cell Type"],
        }
    )
    for receptor in epitopes_all:
        filtered_sampleDF[receptor] = sampleDF[receptor]

    target_receptors = [
        col for col in filtered_sampleDF.columns
        if col not in ["Cell Type", signal_receptor]
    ]
    
    print(f"Analyzing {len(target_receptors)} target receptors: {target_receptors}")
    
    on_target_mask = (filtered_sampleDF["Cell Type"] == targCell).to_numpy()
    off_target_mask = ~on_target_mask
    
    # Store results for all epitopes
    epitope_results = {}
    epitope_flat_regions = {}
    
    palette = sns.color_palette("husl", n_colors=len(target_receptors))
    
    # Analyze each epitope
    for i, receptor in enumerate(target_receptors):
        print(f"Processing {receptor}...")
        
        selectivities = []
        convergence_issues = []
        
        for conv_fact in conversion_factors:
            test_DF = filtered_sampleDF[[signal_receptor, receptor, "Cell Type"]].copy()
            test_DF[receptor] = test_DF[receptor] * conv_fact
            rec_mat = test_DF[[signal_receptor, receptor]].to_numpy()
            
            optSelec, optAffs, converged = optimize_affs(
                targRecs=rec_mat[on_target_mask],
                offTargRecs=rec_mat[off_target_mask],
                dose=dose,
                valencies=model_valencies,
                bounds=affinity_bounds,
            )
            
            if not converged:
                convergence_issues.append(conv_fact)
            
            selectivities.append(1 / optSelec if optSelec > 0 else np.nan)
        
        # Store results
        epitope_results[receptor] = {
            'selectivities': np.array(selectivities),
            'convergence_issues': convergence_issues
        }
        
        # Detect flat regions
        valid_mask = ~np.isnan(selectivities)
        if np.sum(valid_mask) > min_flat_length:
            flat_regions = detect_flat_regions(
                conversion_factors[valid_mask],
                np.array(selectivities)[valid_mask],
                flatness_threshold=flatness_threshold,
                min_flat_length=min_flat_length
            )
            epitope_flat_regions[receptor] = flat_regions
        else:
            epitope_flat_regions[receptor] = []
        
        # Plot selectivity curves with flat regions highlighted
        ax[0].plot(
            conversion_factors, selectivities, 
            marker='o', ls='-', label=receptor, 
            color=palette[i], alpha=0.7
        )
        
        # Highlight flat regions
        for region in epitope_flat_regions[receptor]:
            region_mask = (conversion_factors >= region['start_x']) & (conversion_factors <= region['end_x'])
            ax[0].fill_between(
                conversion_factors[region_mask], 
                np.array(selectivities)[region_mask],
                alpha=0.3, color=palette[i]
            )
        
        # Mark convergence issues
        for conv_issue in convergence_issues:
            ax[0].axvline(x=conv_issue, color='red', alpha=0.3, linewidth=0.5)
    
    # Configure selectivity plot
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlabel('Scaling Factor')
    ax[0].set_ylabel('Selectivity')
    ax[0].set_title('Selectivity vs Scaling Factor (Flat Regions Highlighted)')
    ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax[0].grid(True, alpha=0.3)
    
    # Plot flat region characteristics
    epitope_names = []
    region_starts = []
    region_ends = []
    region_selectivities = []
    
    for epitope, regions in epitope_flat_regions.items():
        for region in regions:
            epitope_names.append(epitope)
            region_starts.append(region['start_x'])
            region_ends.append(region['end_x'])
            region_selectivities.append(region['avg_selectivity'])
    
    if epitope_names:  # Only plot if we have flat regions
        flat_df = pd.DataFrame({
            'Epitope': epitope_names,
            'Start': region_starts,
            'End': region_ends,
            'Selectivity': region_selectivities
        })
        
        # Horizontal bar chart showing flat region ranges
        y_pos = np.arange(len(flat_df))
        ax[1].barh(y_pos, np.log10(flat_df['End']) - np.log10(flat_df['Start']), 
                   left=np.log10(flat_df['Start']), alpha=0.7)
        ax[1].set_yticks(y_pos)
        ax[1].set_yticklabels(flat_df['Epitope'])
        ax[1].set_xlabel('Log10(Scaling Factor)')
        ax[1].set_title('Flat Region Ranges')
        ax[1].grid(True, alpha=0.3)
    
    # Find and plot universal scaling factors
    universal_factors = find_universal_scaling_factors(epitope_flat_regions, conversion_factors)
    
    if universal_factors:
        # Plot number of epitopes in flat region vs scaling factor
        factors = [uf['scaling_factor'] for uf in universal_factors]
        num_epitopes = [uf['num_epitopes'] for uf in universal_factors]
        avg_selectivities = [uf['avg_selectivity'] for uf in universal_factors]
        
        ax[2].scatter(factors, num_epitopes, c=avg_selectivities, 
                     cmap='viridis', s=50, alpha=0.7)
        ax[2].set_xscale('log')
        ax[2].set_xlabel('Scaling Factor')
        ax[2].set_ylabel('Number of Epitopes in Flat Region')
        ax[2].set_title('Universal Scaling Factor Candidates')
        cbar = f.colorbar(ax[2].collections[0], ax=ax[2])
        cbar.set_label('Average Selectivity')
        ax[2].grid(True, alpha=0.3)
        
        # Find best universal factor
        best_factor = max(universal_factors, key=lambda x: (x['num_epitopes'], x['min_selectivity']))
        ax[2].axvline(x=best_factor['scaling_factor'], color='red', linestyle='--', 
                     label=f"Best: {best_factor['scaling_factor']:.1e}")
        ax[2].legend()
        
        print(f"\nBest universal scaling factor: {best_factor['scaling_factor']:.2e}")
        print(f"Covers {best_factor['num_epitopes']} epitopes")
        print(f"Minimum selectivity: {best_factor['min_selectivity']:.2f}")
    
    # Plot gradient analysis for a few representative epitopes
    for i, receptor in enumerate(target_receptors):
        if receptor in epitope_results:
            selectivities = epitope_results[receptor]['selectivities']
            valid_mask = ~np.isnan(selectivities)
            if np.sum(valid_mask) > 3:
                smoothed_sel = gaussian_filter1d(selectivities[valid_mask], sigma=1.0)
                gradients = np.abs(np.gradient(smoothed_sel, conversion_factors[valid_mask]))
                
                ax[3].plot(conversion_factors[valid_mask], gradients, 
                          label=f'{receptor}', color=palette[i])
    
    ax[3].axhline(y=flatness_threshold, color='red', linestyle='--', 
                  label=f'Flatness threshold ({flatness_threshold})')
    ax[3].set_xscale('log')
    ax[3].set_yscale('log')
    ax[3].set_xlabel('Scaling Factor')
    ax[3].set_ylabel('Gradient Magnitude')
    ax[3].set_title('Gradient Analysis (Flatness Detection)')
    ax[3].legend()
    ax[3].grid(True, alpha=0.3)
    
    # Summary statistics table
    summary_text = f"\n\nAnalysis Summary:\n"
    summary_text += f"• Analyzed {len(target_receptors)} epitopes\n"
    summary_text += f"• Found {sum(len(regions) for regions in epitope_flat_regions.values())} flat regions\n"
    if universal_factors:
        summary_text += f"• {len(universal_factors)} universal scaling factors identified\n"
        summary_text += f"• Best factor: {best_factor['scaling_factor']:.2e}\n\n"
    
    print(summary_text)
    
    return f
