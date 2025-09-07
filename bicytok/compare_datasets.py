#!/usr/bin/env python3
"""
Script to compare different scRNA-seq and CITE-seq datasets to ensure consistency
in cell counts and labels across all data sources.

This script compares:
1. Existing CITE-seq data (CITEdata_SurfMarkers.zip)
2. Downloaded ADT data from Hao et al. 2021
3. Downloaded RNA data from Hao et al. 2021  
4. Cell type annotations generated from process_scRNAseq.qmd

Author: Generated for bi-cytok-2 project
"""

import os
import pandas as pd
import numpy as np
import scanpy as sc
from pathlib import Path
import gzip

# Import the existing CITE data loading function
from imports import importCITE

# Set up paths
path_here = Path(__file__).parent
DATA_DIR = path_here / "data"
CUSTOM_ANNOT_DIR = DATA_DIR / "custom_annotations"
ADT_DATA_PATH = "/home/sama/Hao_et_al_CITE-seq_data/ADT_data"
RNA_DATA_PATH = "/home/sama/Hao_et_al_CITE-seq_data/RNA_data"


def load_existing_cite_data():
    """Load the existing CITE-seq surface marker data"""
    print("Loading existing CITE-seq data...")
    try:
        cite_df = importCITE()
        print(f"Existing CITE data shape: {cite_df.shape}")
        print(f"Columns: {list(cite_df.columns[:10])}{'...' if len(cite_df.columns) > 10 else ''}")
        
        # Check for cell type columns
        cell_type_cols = [col for col in cite_df.columns if 'cell' in col.lower() or 'type' in col.lower()]
        print(f"Potential cell type columns: {cell_type_cols}")
        
        return cite_df
    except Exception as e:
        print(f"Error loading existing CITE data: {e}")
        return None


def load_adt_data():
    """Load the downloaded ADT data from Hao et al."""
    print("\nLoading ADT data from Hao et al...")
    try:
        # Load 10x format data
        adata_adt = sc.read_10x_mtx(
            ADT_DATA_PATH,
            prefix="GSM5008738_ADT_3P-",
            cache=False
        )
        adata_adt.var_names_make_unique()
        
        print(f"ADT data shape: {adata_adt.shape}")
        print(f"Number of cells: {adata_adt.n_obs}")
        print(f"Number of features: {adata_adt.n_vars}")
        print(f"First 10 features: {list(adata_adt.var_names[:10])}")
        
        return adata_adt
    except Exception as e:
        print(f"Error loading ADT data: {e}")
        return None


def load_rna_data():
    """Load the downloaded RNA data from Hao et al."""
    print("\nLoading RNA data from Hao et al...")
    try:
        # Load 10x format data
        adata_rna = sc.read_10x_mtx(
            RNA_DATA_PATH,
            prefix="GSM5008737_RNA_3P-",
            cache=False
        )
        adata_rna.var_names_make_unique()
        
        print(f"RNA data shape: {adata_rna.shape}")
        print(f"Number of cells: {adata_rna.n_obs}")
        print(f"Number of features: {adata_rna.n_vars}")
        print(f"First 10 features: {list(adata_rna.var_names[:10])}")
        
        return adata_rna
    except Exception as e:
        print(f"Error loading RNA data: {e}")
        return None


def load_custom_annotations():
    """Load the cell type annotations generated from process_scRNAseq.qmd"""
    print("\nLoading custom cell type annotations...")
    annot_file = CUSTOM_ANNOT_DIR / "cell_labels.tsv"
    
    if not annot_file.exists():
        print(f"Custom annotations file not found: {annot_file}")
        print("Please run process_scRNAseq.qmd first to generate annotations.")
        return None
    
    try:
        annot_df = pd.read_csv(annot_file, sep='\t')
        print(f"Custom annotations shape: {annot_df.shape}")
        print(f"Columns: {list(annot_df.columns)}")
        print(f"Number of unique clusters: {annot_df['leiden_cluster'].nunique()}")
        print(f"Cluster distribution:\n{annot_df['leiden_cluster'].value_counts().sort_index()}")
        
        return annot_df
    except Exception as e:
        print(f"Error loading custom annotations: {e}")
        return None


def compare_cell_barcodes():
    """Compare cell barcodes across ADT and RNA datasets"""
    print("\n" + "="*60)
    print("COMPARING CELL BARCODES")
    print("="*60)
    
    # Load barcode files directly
    try:
        with gzip.open(f"{ADT_DATA_PATH}/GSM5008738_ADT_3P-barcodes.tsv.gz", 'rt') as f:
            adt_barcodes = set(line.strip() for line in f)
        
        with gzip.open(f"{RNA_DATA_PATH}/GSM5008737_RNA_3P-barcodes.tsv.gz", 'rt') as f:
            rna_barcodes = set(line.strip() for line in f)
        
        print(f"ADT barcodes count: {len(adt_barcodes)}")
        print(f"RNA barcodes count: {len(rna_barcodes)}")
        
        # Check overlap
        common_barcodes = adt_barcodes.intersection(rna_barcodes)
        adt_only = adt_barcodes - rna_barcodes
        rna_only = rna_barcodes - adt_barcodes
        
        print(f"Common barcodes: {len(common_barcodes)}")
        print(f"ADT-only barcodes: {len(adt_only)}")
        print(f"RNA-only barcodes: {len(rna_only)}")
        
        if len(common_barcodes) > 0:
            print("✓ Datasets share common cell barcodes")
        else:
            print("✗ No common cell barcodes found!")
            
        return {
            'adt_barcodes': adt_barcodes,
            'rna_barcodes': rna_barcodes,
            'common_barcodes': common_barcodes
        }
        
    except Exception as e:
        print(f"Error comparing barcodes: {e}")
        return None


def main():
    """Main comparison function"""
    print("DATASET COMPARISON REPORT")
    print("="*60)
    
    # Load all datasets
    cite_df = load_existing_cite_data()
    adt_data = load_adt_data()
    rna_data = load_rna_data()
    annot_df = load_custom_annotations()
    
    # Compare cell barcodes
    barcode_comparison = compare_cell_barcodes()
    
    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    
    datasets = []
    if cite_df is not None:
        datasets.append(("Existing CITE data", cite_df.shape[0]))
    if adt_data is not None:
        datasets.append(("Downloaded ADT data", adt_data.n_obs))
    if rna_data is not None:
        datasets.append(("Downloaded RNA data", rna_data.n_obs))
    if annot_df is not None:
        datasets.append(("Custom annotations", annot_df.shape[0]))
    
    # Print summary table
    print("\nCell count summary:")
    print("-" * 40)
    for name, count in datasets:
        print(f"{name:<25}: {count:>10,}")
    
    # Check if all counts match
    if datasets:
        counts = [count for _, count in datasets]
        if len(set(counts)) == 1:
            print(f"\n✓ All datasets have matching cell counts: {counts[0]:,}")
        else:
            print(f"\n✗ Cell counts do not match across datasets!")
            print(f"  Unique counts: {sorted(set(counts))}")
    
    # Additional checks for barcode consistency
    if barcode_comparison and adt_data and rna_data:
        print(f"\nBarcode consistency:")
        print(f"- ADT and RNA share {len(barcode_comparison['common_barcodes']):,} common barcodes")
        if annot_df is not None:
            # Check if annotation barcodes match RNA/ADT barcodes
            annot_barcodes = set(annot_df['barcode'].values)
            rna_vs_annot = barcode_comparison['rna_barcodes'].intersection(annot_barcodes)
            print(f"- RNA and annotations share {len(rna_vs_annot):,} common barcodes")
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)


def compare_cell_type_labels():
    """Compare cell type labels between existing CITE data and Hao et al metadata."""
    print("\n" + "="*60)
    print("CELL TYPE LABELS COMPARISON")
    print("="*60)
    
    try:
        # Load existing CITE data with cell type labels
        cite_path = "bicytok/data/CITEdata_SurfMarkers.zip"
        cite_df = pd.read_csv(cite_path)
        
        # Load Hao et al metadata
        meta_file = "/home/sama/Hao_et_al_CITE-seq_data/meta_data/GSE164378_sc.meta.data_3P.csv.gz"
        if Path(meta_file).exists():
            hao_meta = pd.read_csv(meta_file, compression='gzip')
            print(f"✓ Successfully loaded Hao et al metadata: {hao_meta.shape}")
        else:
            print(f"✗ Metadata file not found: {meta_file}")
            return False
        
        print(f"\nDataset shapes:")
        print(f"  Existing CITE data: {cite_df.shape}")
        print(f"  Hao et al metadata: {hao_meta.shape}")
        
        # Check if cell counts match
        if cite_df.shape[0] == hao_meta.shape[0]:
            print(f"✓ Cell counts match: {cite_df.shape[0]:,} cells")
        else:
            print(f"✗ Cell counts differ: CITE={cite_df.shape[0]:,}, Hao={hao_meta.shape[0]:,}")
        
        # Compare cell type hierarchies
        print(f"\nCell type hierarchy comparison:")
        print("-" * 40)
        
        # Level 1 comparison
        cite_l1 = set(cite_df['CellType1'].unique())
        hao_l1 = set(hao_meta['celltype.l1'].unique())
        
        print(f"Level 1 (CellType1 vs celltype.l1):")
        print(f"  CITE unique types: {len(cite_l1)} - {sorted(cite_l1)}")
        print(f"  Hao unique types:  {len(hao_l1)} - {sorted(hao_l1)}")
        print(f"  Common types: {sorted(cite_l1.intersection(hao_l1))}")
        print(f"  Only in CITE: {sorted(cite_l1 - hao_l1)}")
        print(f"  Only in Hao: {sorted(hao_l1 - cite_l1)}")
        
        # Level 2 comparison
        cite_l2 = set(cite_df['CellType2'].unique())
        hao_l2 = set(hao_meta['celltype.l2'].unique())
        
        print(f"\nLevel 2 (CellType2 vs celltype.l2):")
        print(f"  CITE unique types: {len(cite_l2)}")
        print(f"  Hao unique types:  {len(hao_l2)}")
        print(f"  Common types: {len(cite_l2.intersection(hao_l2))}")
        
        # Level 3 comparison
        cite_l3 = set(cite_df['CellType3'].unique())
        hao_l3 = set(hao_meta['celltype.l3'].unique())
        
        print(f"\nLevel 3 (CellType3 vs celltype.l3):")
        print(f"  CITE unique types: {len(cite_l3)}")
        print(f"  Hao unique types:  {len(hao_l3)}")
        print(f"  Common types: {len(cite_l3.intersection(hao_l3))}")
        
        # Check barcode alignment if possible
        print(f"\nBarcode format comparison:")
        print(f"  CITE Cell IDs (first 3): {list(cite_df['Cell'].head(3))}")
        print(f"  Hao barcodes (first 3): {list(hao_meta.iloc[:3, 0])}")
        
        # Check if the datasets are identical (same cell type labels in same order)
        level_matches = []
        if len(cite_l1.symmetric_difference(hao_l1)) == 0:
            level_matches.append("Level 1: ✓ IDENTICAL")
        else:
            level_matches.append("Level 1: ✗ DIFFERENT")
            
        if len(cite_l2.symmetric_difference(hao_l2)) == 0:
            level_matches.append("Level 2: ✓ IDENTICAL") 
        else:
            level_matches.append("Level 2: ✗ DIFFERENT")
            
        if len(cite_l3.symmetric_difference(hao_l3)) == 0:
            level_matches.append("Level 3: ✓ IDENTICAL")
        else:
            level_matches.append("Level 3: ✗ DIFFERENT")
        
        print(f"\nSummary:")
        for match in level_matches:
            print(f"  {match}")
        
        return True
        
    except Exception as e:
        print(f"Error in cell type comparison: {e}")
        return False


def compare_adt_expression_values():
    """Compare ADT expression values between existing CITE data and Hao et al ADT data."""
    print("\n" + "="*60)
    print("ADT EXPRESSION VALUES COMPARISON")
    print("="*60)
    
    try:
        # Load existing CITE data
        cite_path = "bicytok/data/CITEdata_SurfMarkers.zip"
        cite_df = pd.read_csv(cite_path)
        print(f"Existing CITE data shape: {cite_df.shape}")
        
        # Load Hao et al ADT data
        adt_dir = "/home/sama/Hao_et_al_CITE-seq_data/ADT_data/"
        adt_data = sc.read_10x_mtx(
            adt_dir,
            prefix="GSM5008738_ADT_3P-",
            cache=False
        )
        adt_data.var_names_make_unique()
        print(f"Hao et al ADT data shape: {adt_data.shape}")
        
        # Get overlapping markers
        cite_markers = set(cite_df.columns[1:])  # Skip 'Cell' column
        hao_markers = set(adt_data.var_names)
        common_markers = cite_markers.intersection(hao_markers)
        
        print(f"\nMarker comparison:")
        print(f"  CITE markers: {len(cite_markers)}")
        print(f"  Hao markers: {len(hao_markers)}")
        print(f"  Common markers: {len(common_markers)}")
        print(f"  Coverage: {len(common_markers)/len(cite_markers)*100:.1f}% of CITE markers")
        
        if len(common_markers) > 0:
            print(f"  Sample common markers: {sorted(list(common_markers))[:5]}")
            
            # Compare distributions for sample markers
            sample_markers = sorted(list(common_markers))[:5]
            print(f"\nExpression value comparison for: {sample_markers}")
            print("-" * 60)
            
            all_correlations = []
            for marker in sample_markers:
                if marker in cite_df.columns:
                    cite_values = cite_df[marker].values
                    hao_values = adt_data[:, marker].X.toarray().flatten()
                    
                    print(f"\n{marker}:")
                    print(f"  CITE - Range: [{cite_values.min():.2f}, {cite_values.max():.2f}], "
                          f"Mean: {cite_values.mean():.2f}, Std: {cite_values.std():.2f}")
                    print(f"  Hao  - Range: [{hao_values.min():.2f}, {hao_values.max():.2f}], "
                          f"Mean: {hao_values.mean():.2f}, Std: {hao_values.std():.2f}")
                    
                    # Calculate correlation for first 1000 cells
                    if len(cite_values) >= 1000 and len(hao_values) >= 1000:
                        correlation = pd.Series(cite_values[:1000]).corr(pd.Series(hao_values[:1000]))
                        all_correlations.append(correlation)
                        print(f"  Correlation (first 1000 cells): {correlation:.3f}")
                        
                        # Check if values appear to be raw counts vs normalized
                        cite_has_floats = (cite_values % 1 != 0).any()
                        hao_has_floats = (hao_values % 1 != 0).any()
                        print(f"  CITE has non-integer values: {cite_has_floats}")
                        print(f"  Hao has non-integer values: {hao_has_floats}")
            
            # Overall assessment
            if all_correlations:
                mean_correlation = pd.Series(all_correlations).mean()
                print(f"\nOverall Assessment:")
                print(f"  Average correlation: {mean_correlation:.3f}")
                
                if mean_correlation > 0.95:
                    print("  ✓ IDENTICAL: Expression values are highly correlated (likely same processing)")
                elif mean_correlation > 0.8:
                    print("  ~ SIMILAR: Expression values are well correlated (possibly different normalization)")
                elif mean_correlation > 0.5:
                    print("  ? MODERATE: Expression values are moderately correlated (different processing)")
                else:
                    print("  ✗ DIFFERENT: Expression values are poorly correlated (different data/processing)")
                    
        else:
            print("\nNo common markers found - checking marker naming:")
            print(f"  CITE sample markers: {sorted(list(cite_markers))[:10]}")
            print(f"  Hao sample markers: {sorted(list(hao_markers))[:10]}")
            print("\n⚠️  Marker names may differ between datasets")
        
        return True
        
    except Exception as e:
        print(f"Error in ADT expression comparison: {e}")
        return False


if __name__ == "__main__":
    main()
    compare_cell_type_labels()
    compare_adt_expression_values()
