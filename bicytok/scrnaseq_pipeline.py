"""Modular scRNA-seq processing pipeline for Leiden clustering analysis"""

import os

import pandas as pd
import scanpy as sc

# Constants
DEFAULT_DATA_PATH = "/home/sama/Hao_et_al_CITE-seq_data/RNA_data"
DEFAULT_PREFIX = "GSM5008737_RNA_3P-"
DEFAULT_TARGET_SUM = 1e4
DEFAULT_N_COMPONENTS = 50
DEFAULT_N_NEIGHBORS = 15
DEFAULT_N_PCS = 9


def load_10x_data(
    data_path: str = DEFAULT_DATA_PATH, prefix: str = DEFAULT_PREFIX
) -> sc.AnnData:
    """
    Load 10X Genomics data and make variable names unique.

    Parameters
    ----------
    data_path : str
        Path to the directory containing 10X data files
    prefix : str
        Prefix for the 10X files

    Returns
    -------
    sc.AnnData
        Raw AnnData object with unique variable names
    """
    adata = sc.read_10x_mtx(data_path, prefix=prefix)
    adata.var_names_make_unique()
    return adata


def calculate_cell_metrics(adata: sc.AnnData) -> sc.AnnData:
    """
    Calculate per-cell quality metrics.

    Parameters
    ----------
    adata : sc.AnnData
        Input AnnData object

    Returns
    -------
    sc.AnnData
        AnnData object with added cell metrics in .obs
    """
    adata.obs["n_genes"] = (adata.X > 0).sum(axis=1)
    adata.obs["total_counts"] = adata.X.sum(axis=1)
    return adata


def normalize_data(
    adata: sc.AnnData, target_sum: float = DEFAULT_TARGET_SUM
) -> sc.AnnData:
    """
    Normalize and log-transform the data.

    Parameters
    ----------
    adata : sc.AnnData
        Input AnnData object
    target_sum : float
        Target sum for normalization

    Returns
    -------
    sc.AnnData
        Normalized and log-transformed AnnData object
    """
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    return adata


def find_highly_variable_genes(
    adata: sc.AnnData,
    min_mean: float = 0.0005,
    max_mean: float = 10,
    min_disp: float = 0.5,
) -> sc.AnnData:
    """
    Identify highly variable genes and subset the data.

    Parameters
    ----------
    adata : sc.AnnData
        Input AnnData object
    min_mean : float
        Minimum mean expression threshold
    max_mean : float
        Maximum mean expression threshold
    min_disp : float
        Minimum dispersion threshold

    Returns
    -------
    sc.AnnData
        AnnData object subset to highly variable genes
    """
    sc.pp.highly_variable_genes(
        adata,
        min_mean=min_mean,
        max_mean=max_mean,
        min_disp=min_disp,
        flavor="seurat_v3",
    )
    adata_hvg = adata[:, adata.var.highly_variable].copy()
    return adata_hvg


def scale_data(adata: sc.AnnData, max_value: float = 10) -> sc.AnnData:
    """
    Z-score normalize the data.

    Parameters
    ----------
    adata : sc.AnnData
        Input AnnData object
    max_value : float
        Maximum value after scaling

    Returns
    -------
    sc.AnnData
        Scaled AnnData object
    """
    sc.pp.scale(adata, max_value=max_value)
    return adata


def perform_pca(adata: sc.AnnData, n_comps: int = DEFAULT_N_COMPONENTS) -> sc.AnnData:
    """
    Perform Principal Component Analysis.

    Parameters
    ----------
    adata : sc.AnnData
        Input AnnData object
    n_comps : int
        Number of principal components to compute

    Returns
    -------
    sc.AnnData
        AnnData object with PCA results
    """
    sc.tl.pca(adata, n_comps=n_comps)
    return adata


def build_neighborhood_graph(
    adata: sc.AnnData,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    n_pcs: int = DEFAULT_N_PCS,
) -> sc.AnnData:
    """
    Compute the neighborhood graph for clustering.

    Parameters
    ----------
    adata : sc.AnnData
        Input AnnData object with PCA results
    n_neighbors : int
        Number of neighbors for the graph
    n_pcs : int
        Number of principal components to use

    Returns
    -------
    sc.AnnData
        AnnData object with neighborhood graph
    """
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    return adata


def perform_leiden_clustering(
    adata: sc.AnnData, resolution: float = 1.5, n_iterations: int = 2
) -> sc.AnnData:
    """
    Perform Leiden clustering.

    Parameters
    ----------
    adata : sc.AnnData
        Input AnnData object with neighborhood graph
    resolution : float
        Resolution parameter for Leiden clustering
    n_iterations : int
        Number of iterations for the Leiden algorithm

    Returns
    -------
    sc.AnnData
        AnnData object with Leiden cluster assignments
    """
    sc.tl.leiden(
        adata,
        resolution=resolution,
        flavor="igraph",
        n_iterations=n_iterations,
        directed=False,
    )
    return adata


def compute_umap(adata: sc.AnnData) -> sc.AnnData:
    """
    Compute UMAP embedding for visualization.

    Parameters
    ----------
    adata : sc.AnnData
        Input AnnData object with neighborhood graph

    Returns
    -------
    sc.AnnData
        AnnData object with UMAP coordinates
    """
    sc.tl.umap(adata)
    return adata


def find_marker_genes(adata: sc.AnnData, groupby: str = "leiden") -> sc.AnnData:
    """
    Find marker genes for each cluster.

    Parameters
    ----------
    adata : sc.AnnData
        Input AnnData object with cluster assignments
    groupby : str
        Column name in .obs to group by for marker gene analysis

    Returns
    -------
    sc.AnnData
        AnnData object with marker gene results
    """
    sc.tl.rank_genes_groups(adata, groupby, method="wilcoxon")
    return adata


def get_cell_type_markers() -> dict[str, list]:
    """
    Get the dictionary of cell type marker genes from literature.

    Returns
    -------
    Dict[str, list]
        Dictionary mapping cell type names to lists of marker genes
    """
    marker_dict = {
        "B intermediate": [
            "MS4A1",
            "TNFRSF13B",
            "IGHM",
            "IGHD",
            "AIM2",
            "CD79A",
            "LINC01857",
            "RALGPS2",
            "BANK1",
            "CD79B",
        ],
        "B memory": [
            "MS4A1",
            "COCH",
            "AIM2",
            "BANK1",
            "SSPN",
            "CD79A",
            "TEX9",
            "RALGPS2",
            "TNFRSF13C",
            "LINC01781",
        ],
        "B naive": [
            "IGHM",
            "IGHD",
            "CD79A",
            "IL4R",
            "MS4A1",
            "CXCR4",
            "BTG1",
            "TCL1A",
            "CD79B",
            "YBX3",
        ],
        "Plasmablast": [
            "IGHA2",
            "MZB1",
            "TNFRSF17",
            "DERL3",
            "TXNDC5",
            "TNFRSF13B",
            "POU2AF1",
            "CPNE5",
            "HRASLS2",
            "NT5DC2",
        ],
        "CD4 CTL": [
            "GZMH",
            "CD4",
            "FGFBP2",
            "ITGB1",
            "GZMA",
            "CST7",
            "GNLY",
            "B2M",
            "IL32",
            "NKG7",
        ],
        "CD4 Naive": [
            "TCF7",
            "CD4",
            "CCR7",
            "IL7R",
            "FHIT",
            "LEF1",
            "MAL",
            "NOSIP",
            "LDHB",
            "PIK3IP1",
        ],
        "CD4 Proliferating": [
            "MKI67",
            "TOP2A",
            "PCLAF",
            "CENPF",
            "TYMS",
            "NUSAP1",
            "ASPM",
            "PTTG1",
            "TPX2",
            "RRM2",
        ],
        "CD4 TCM": [
            "IL7R",
            "TMSB10",
            "CD4",
            "ITGB1",
            "LTB",
            "TRAC",
            "AQP3",
            "LDHB",
            "IL32",
            "MAL",
        ],
        "CD4 TEM": [
            "IL7R",
            "CCL5",
            "FYB1",
            "GZMK",
            "IL32",
            "GZMA",
            "KLRB1",
            "TRAC",
            "LTB",
            "AQP3",
        ],
        "Treg": [
            "RTKN2",
            "FOXP3",
            "AC133644.2",
            "CD4",
            "IL2RA",
            "TIGIT",
            "CTLA4",
            "FCRL3",
            "LAIR2",
            "IKZF2",
        ],
        "CD8 Naive": [
            "CD8B",
            "S100B",
            "CCR7",
            "RGS10",
            "NOSIP",
            "LINC02446",
            "LEF1",
            "CRTAM",
            "CD8A",
            "OXNAD1",
        ],
        "CD8 Proliferating": [
            "MKI67",
            "CD8B",
            "TYMS",
            "TRAC",
            "PCLAF",
            "CD3D",
            "CLSPN",
            "CD3G",
            "TK1",
            "RRM2",
        ],
        "CD8 TCM": [
            "CD8B",
            "ANXA1",
            "CD8A",
            "KRT1",
            "LINC02446",
            "YBX3",
            "IL7R",
            "TRAC",
            "NELL2",
            "LDHB",
        ],
        "CD8 TEM": [
            "CCL5",
            "GZMH",
            "CD8A",
            "TRAC",
            "KLRD1",
            "NKG7",
            "GZMK",
            "CST7",
            "CD8B",
            "TRGC2",
        ],
        "ASDC": [
            "PPP1R14A",
            "LILRA4",
            "AXL",
            "IL3RA",
            "SCT",
            "SCN9A",
            "LGMN",
            "DNASE1L3",
            "CLEC4C",
            "GAS6",
        ],
        "cDC1": [
            "CLEC9A",
            "DNASE1L3",
            "C1orf54",
            "IDO1",
            "CLNK",
            "CADM1",
            "FLT3",
            "ENPP1",
            "XCR1",
            "NDRG2",
        ],
        "cDC2": [
            "FCER1A",
            "HLA-DQA1",
            "CLEC10A",
            "CD1C",
            "ENHO",
            "PLD4",
            "GSN",
            "SLC38A1",
            "NDRG2",
            "AFF3",
        ],
        "pDC": [
            "ITM2C",
            "PLD4",
            "SERPINF1",
            "LILRA4",
            "IL3RA",
            "TPM2",
            "MZB1",
            "SPIB",
            "IRF4",
            "SMPD3",
        ],
        "CD14 Mono": [
            "S100A9",
            "CTSS",
            "S100A8",
            "LYZ",
            "VCAN",
            "S100A12",
            "IL1B",
            "CD14",
            "G0S2",
            "FCN1",
        ],
        "CD16 Mono": [
            "CDKN1C",
            "FCGR3A",
            "PTPRC",
            "LST1",
            "IER5",
            "MS4A7",
            "RHOC",
            "IFITM3",
            "AIF1",
            "HES4",
        ],
        "NK": [
            "GNLY",
            "TYROBP",
            "NKG7",
            "FCER1G",
            "GZMB",
            "TRDC",
            "PRF1",
            "FGFBP2",
            "SPON2",
            "KLRF1",
        ],
        "NK Proliferating": [
            "MKI67",
            "KLRF1",
            "TYMS",
            "TRDC",
            "TOP2A",
            "FCER1G",
            "PCLAF",
            "CD247",
            "CLSPN",
            "ASPM",
        ],
        "NK_CD56bright": [
            "XCL2",
            "FCER1G",
            "SPINK2",
            "TRDC",
            "KLRC1",
            "XCL1",
            "SPTSSB",
            "PPP1R9A",
            "NCAM1",
            "TNFRSF11A",
        ],
        "Eryth": [
            "HBD",
            "HBM",
            "AHSP",
            "ALAS2",
            "CA1",
            "SLC4A1",
            "IFIT1B",
            "TRIM58",
            "SELENBP1",
            "TMCC2",
        ],
        "HSPC": [
            "SPINK2",
            "PRSS57",
            "CYTL1",
            "EGFL7",
            "GATA2",
            "CD34",
            "SMIM24",
            "AVP",
            "MYB",
            "LAPTM4B",
        ],
        "ILC": [
            "KIT",
            "TRDC",
            "TTLL10",
            "LINC01229",
            "SOX4",
            "KLRB1",
            "TNFRSF18",
            "TNFRSF4",
            "IL1R1",
            "HPGDS",
        ],
        "Platelet": [
            "PPBP",
            "PF4",
            "NRGN",
            "GNG11",
            "CAVIN2",
            "TUBB1",
            "CLU",
            "HIST1H2AC",
            "RGS18",
            "GP9",
        ],
        "dnT": [
            "PTPN3",
            "MIR4422HG",
            "NUCB2",
            "CAV1",
            "DTHD1",
            "GZMA",
            "MYB",
            "FXYD2",
            "GZMK",
            "AC004585.1",
        ],
        "gdT": [
            "TRDC",
            "TRGC1",
            "TRGC2",
            "KLRC1",
            "NKG7",
            "TRDV2",
            "CD7",
            "TRGV9",
            "KLRD1",
            "KLRG1",
        ],
        "MAIT": [
            "KLRB1",
            "NKG7",
            "GZMK",
            "IL7R",
            "SLC4A10",
            "GZMA",
            "CXCR6",
            "PRSS35",
            "RBM24",
            "NCR3",
        ],
    }
    return marker_dict


def calculate_marker_overlap(
    adata: sc.AnnData, marker_dict: dict[str, list] | None = None
) -> pd.DataFrame:
    """
    Calculate overlap between cluster markers and known cell type markers.

    Parameters
    ----------
    adata : sc.AnnData
        Input AnnData object with marker gene results
    marker_dict : Dict[str, list], optional
        Dictionary of cell type markers. If None, uses default markers.

    Returns
    -------
    pd.DataFrame
        DataFrame showing marker gene overlap scores
    """
    if marker_dict is None:
        marker_dict = get_cell_type_markers()

    return sc.tl.marker_gene_overlap(adata, marker_dict)


def run_scrnaseq_pipeline(
    leiden_resolution: float = 1.5,
    data_path: str = DEFAULT_DATA_PATH,
    prefix: str = DEFAULT_PREFIX,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    n_pcs: int = DEFAULT_N_PCS,
    n_components: int = DEFAULT_N_COMPONENTS,
    target_sum: float = DEFAULT_TARGET_SUM,
    return_full_adata: bool = False,
) -> pd.Series:
    """
    Run the complete scRNA-seq processing pipeline and return Leiden cluster assignments.

    Parameters
    ----------
    leiden_resolution : float
        Resolution parameter for Leiden clustering
    data_path : str
        Path to the directory containing 10X data files
    prefix : str
        Prefix for the 10X files
    n_neighbors : int
        Number of neighbors for the neighborhood graph
    n_pcs : int
        Number of principal components to use for neighborhood graph
    n_components : int
        Number of principal components to compute
    target_sum : float
        Target sum for normalization
    return_full_adata : bool
        If True, return the full processed AnnData object instead of just cluster assignments

    Returns
    -------
    pd.Series or sc.AnnData
        If return_full_adata is False (default): Series with cell barcodes as index
        and Leiden cluster assignments as values.
        If return_full_adata is True: Full processed AnnData object.
    """
    # Load data
    print("Loading 10X data...")
    adata = load_10x_data(data_path, prefix)
    print(f"Loaded {adata.n_obs} cells and {adata.n_vars} genes")

    # Calculate cell metrics
    print("Calculating cell metrics...")
    adata = calculate_cell_metrics(adata)

    # Normalize data
    print("Normalizing data...")
    adata = normalize_data(adata, target_sum)

    # Find highly variable genes
    print("Finding highly variable genes...")
    adata_hvg = find_highly_variable_genes(adata)
    print(f"Selected {adata_hvg.n_vars} highly variable genes")

    # Scale data
    print("Scaling data...")
    adata_hvg = scale_data(adata_hvg)

    # Perform PCA
    print(f"Performing PCA with {n_components} components...")
    adata_hvg = perform_pca(adata_hvg, n_components)

    # Build neighborhood graph
    print(
        f"Building neighborhood graph with {n_neighbors} neighbors and {n_pcs} PCs..."
    )
    adata_hvg = build_neighborhood_graph(adata_hvg, n_neighbors, n_pcs)

    # Perform Leiden clustering
    print(f"Performing Leiden clustering with resolution {leiden_resolution}...")
    adata_hvg = perform_leiden_clustering(adata_hvg, leiden_resolution)
    n_clusters = len(adata_hvg.obs["leiden"].unique())
    print(f"Found {n_clusters} clusters")

    # Compute UMAP for visualization
    print("Computing UMAP...")
    adata_hvg = compute_umap(adata_hvg)

    # Find marker genes
    print("Finding marker genes...")
    adata_hvg = find_marker_genes(adata_hvg)

    if return_full_adata:
        return adata_hvg
    else:
        # Return cluster assignments as a pandas Series
        cluster_assignments = pd.Series(
            adata_hvg.obs["leiden"].astype(str).values,
            index=adata_hvg.obs_names,
            name="leiden_cluster",
        )
        return cluster_assignments


def export_cluster_assignments(
    cluster_assignments: pd.Series,
    output_dir: str = "/home/sama/bi-cytok-2/bicytok/data/custom_annotations",
    filename: str = "cell_labels.tsv",
) -> str:
    """
    Export cluster assignments to a TSV file.

    Parameters
    ----------
    cluster_assignments : pd.Series
        Series with cell barcodes as index and cluster assignments as values
    output_dir : str
        Directory to save the output file
    filename : str
        Name of the output file

    Returns
    -------
    str
        Path to the exported file
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create DataFrame for export
    export_df = pd.DataFrame(
        {
            "barcode": cluster_assignments.index,
            "leiden_cluster": cluster_assignments.values,
        }
    )

    # Export to file
    output_path = os.path.join(output_dir, filename)
    export_df.to_csv(output_path, sep="\t", index=False)

    print(f"Exported {len(export_df)} cell labels to {output_path}")
    return output_path


if __name__ == "__main__":
    # Example usage
    print("Running scRNA-seq pipeline with default parameters...")
    cluster_assignments = run_scrnaseq_pipeline(leiden_resolution=1.5)
    print(f"Cluster assignments shape: {cluster_assignments.shape}")
    print(f"Number of unique clusters: {cluster_assignments.nunique()}")
    print("\nFirst 10 cluster assignments:")
    print(cluster_assignments.head(10))

    # Export results
    export_path = export_cluster_assignments(cluster_assignments)
    print(f"Results exported to: {export_path}")
