"""Automated cell type annotation using SCSA with repository and package management"""

import subprocess
import sys
import os
import shutil
import pandas as pd
import tempfile
from pathlib import Path

# Constants
SCSA_REPO_URL = "https://github.com/bioinfo-ibms-pumc/SCSA.git"
TEMP_DIR = "/tmp/scsa_temp"
PROCESSED_DATA_DIR = "/home/sama/bi-cytok-2/bicytok/processed_data"
OUTPUT_DIR = "/home/sama/bi-cytok-2/bicytok/results"

def run_command(command, cwd=None):
    """Execute shell command and handle errors"""
    try:
        result = subprocess.run(command, shell=True, cwd=cwd, 
                              capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {command}")
        print(f"Error: {e.stderr}")
        raise

def get_installed_packages():
    """Get list of currently installed packages"""
    try:
        result = run_command(f"{sys.executable} -m pip list --format=freeze")
        installed = set()
        for line in result.strip().split('\n'):
            if '==' in line:
                package_name = line.split('==')[0]
                installed.add(package_name)
        return installed
    except Exception as e:
        print(f"Warning: Could not get installed packages list: {e}")
        return set()

def install_packages():
    """Install required packages for SCSA"""
    REQUIRED_PACKAGES = [
        "scipy",
        "scikit-learn", 
        "matplotlib",
        "seaborn",
        "plotly"
    ]
    
    pre_existing_packages = get_installed_packages()
    packages_to_install = []
    
    print("Installing required packages...")
    for package in REQUIRED_PACKAGES:
        if package not in pre_existing_packages:
            run_command(f"{sys.executable} -m pip install {package}")
            packages_to_install.append(package)
        else:
            print(f"Package {package} already installed, skipping...")
    
    return packages_to_install

def uninstall_packages(packages_to_uninstall):
    """Uninstall only packages that were installed by the script"""
    if not packages_to_uninstall:
        print("No packages to uninstall.")
        return
        
    print("Uninstalling newly installed packages...")
    for package in packages_to_uninstall:
        try:
            run_command(f"{sys.executable} -m pip uninstall -y {package}")
        except Exception as e:
            print(f"Warning: Could not uninstall {package}: {e}")

def clone_scsa_repository():
    """Clone SCSA repository to temporary directory"""
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    
    print("Cloning SCSA repository...")
    run_command(f"git clone {SCSA_REPO_URL} {TEMP_DIR}")
    
    return TEMP_DIR

def prepare_input_data():
    """Prepare processed input data for SCSA annotation"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Use processed data files from the Quarto notebook
    matrix_file = f"{PROCESSED_DATA_DIR}/scsa_matrix.mtx"
    features_file = f"{PROCESSED_DATA_DIR}/scsa_features.tsv"
    barcodes_file = f"{PROCESSED_DATA_DIR}/scsa_barcodes.tsv"
    
    # Verify processed data files exist
    for file_path in [matrix_file, features_file, barcodes_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Processed data file not found: {file_path}. Run process_scRNAseq.qmd first.")
    
    scsa_input_dir = f"{TEMP_DIR}/input_data"
    os.makedirs(scsa_input_dir, exist_ok=True)
    
    # Copy processed files with standard names expected by SCSA
    shutil.copy(matrix_file, f"{scsa_input_dir}/matrix.mtx")
    shutil.copy(features_file, f"{scsa_input_dir}/features.tsv")
    shutil.copy(barcodes_file, f"{scsa_input_dir}/barcodes.tsv")
    
    return scsa_input_dir

def run_scsa_annotation(input_dir):
    """Run SCSA cell type annotation"""
    print("Running SCSA annotation...")
    
    output_dir = f"{TEMP_DIR}/output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run SCSA annotation command for human data
    scsa_command = f"""
    python SCSA.py \\
        --input {input_dir} \\
        --output {output_dir} \\
        --database CellMarker \\
        --species human \\
        --tissue PBMC \\
        --cluster leiden \\
        --enhanced True
    """
    
    run_command(scsa_command, cwd=TEMP_DIR)
    
    return output_dir

def process_scsa_results(output_dir):
    """Process SCSA results and create cell type annotation dataframe"""
    print("Processing SCSA results...")
    
    # Load SCSA results
    results_file = f"{output_dir}/cluster_celltype.txt"
    
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"SCSA results file not found: {results_file}")
    
    # Read results
    cell_type_df = pd.read_csv(results_file, sep='\t')
    
    # Save processed results
    output_file = f"{OUTPUT_DIR}/scsa_cell_type_annotations.csv"
    cell_type_df.to_csv(output_file, index=False)
    
    print(f"Cell type annotations saved to: {output_file}")
    
    return cell_type_df

def cleanup_repository():
    """Remove SCSA repository and temporary files"""
    print("Cleaning up temporary files...")
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

def main():
    """Main function to run SCSA annotation pipeline"""
    newly_installed_packages = []
    
    try:
        # Step 1: Install required packages
        newly_installed_packages = install_packages()
        
        # Step 2: Clone SCSA repository
        repo_dir = clone_scsa_repository()
        
        # Step 3: Prepare input data
        input_dir = prepare_input_data()
        
        # Step 4: Run SCSA annotation
        output_dir = run_scsa_annotation(input_dir)
        
        # Step 5: Process results
        cell_type_df = process_scsa_results(output_dir)
        
        print("\nSCSA annotation completed successfully!")
        print(f"Annotated {len(cell_type_df)} clusters")
        print("\nCell type summary:")
        print(cell_type_df.head(10))
        
    except Exception as e:
        print(f"Error during SCSA annotation: {e}")
        raise
    
    finally:
        # Cleanup
        cleanup_repository()
        uninstall_packages(newly_installed_packages)
        print("Cleanup completed.")

if __name__ == "__main__":
    main()
