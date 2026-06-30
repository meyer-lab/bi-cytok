"""
Processing and export functions supporting transcript-based scRNA-seq cell type
annotation. All user-specified parameters (cluster assignments, cell types of interest,
marker genes, annotations) are defined in the corresponding figure file:
transcript-based_annotation.qmd.
"""

import pandas as pd
import scanpy as sc

OBS_CARRY_COLS = ["leiden", "pre_annotated_level1", "pre_annotated_level2"]


def load_scRNAseq_data(data_path: str, prefix: str) -> sc.AnnData:
    """
    Loads 10X Genomics data and makes variable names unique.
    """
    adata = sc.read_10x_mtx(data_path, prefix=prefix)
    adata.var_names_make_unique()
    return adata


def compute_cell_metrics(adata: sc.AnnData) -> sc.AnnData:
    """
    Adds per-cell number of detected genes and total counts to .obs.
    """
    adata.obs["n_genes"] = (adata.X > 0).sum(axis=1)
    adata.obs["total_counts"] = adata.X.sum(axis=1)
    return adata


def normalize_and_log(adata: sc.AnnData) -> sc.AnnData:
    """
    Library-size normalizes and log1p-transforms the counts in place.
    """
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    return adata


def select_hvgs(
    adata: sc.AnnData,
    min_mean: float,
    max_mean: float,
    min_disp: float,
) -> sc.AnnData:
    """
    Selects highly variable genes, stashes the log-normalized values in .raw, and
    z-scores the result as PCA input. The values stashed in .raw let downstream marker
    ranking and dot plots read expression values, not z-scores.
    """
    sc.pp.highly_variable_genes(
        adata, min_mean=min_mean, max_mean=max_mean, min_disp=min_disp
    )
    adata_hvg = adata[:, adata.var.highly_variable].copy()
    adata_hvg.raw = adata_hvg
    return adata_hvg


def run_pca(adata: sc.AnnData, n_comps: int) -> sc.AnnData:
    """
    Runs PCA on the input data.
    """
    sc.pp.scale(adata)
    sc.tl.pca(adata, n_comps=n_comps)
    return adata


def build_neighbors(adata: sc.AnnData, n_pcs: int, n_neighbors: int) -> sc.AnnData:
    """
    Computes the neighborhood graph from the first n_pcs principal components.
    """
    sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
    return adata


def cluster_leiden(
    adata: sc.AnnData,
    resolution: float,
    key_added: str,
    random_state: int = 42,
    n_iterations: int = 2,
) -> sc.AnnData:
    """
    Performs Leiden clustering, storing labels in .obs[key_added].
    """
    sc.tl.leiden(
        adata,
        random_state=random_state,
        resolution=resolution,
        flavor="igraph",
        n_iterations=n_iterations,
        key_added=key_added,
    )
    return adata


def compute_umap(adata: sc.AnnData) -> sc.AnnData:
    """
    Computes the UMAP embedding for visualization.
    """
    sc.tl.umap(adata)
    return adata


def add_reference_annotations(
    adata: sc.AnnData, reference_labels: dict[str, pd.Series]
) -> sc.AnnData:
    """
    Attaches pre-annotated reference cell type labels to .obs, aligning each label
    series by position with the cells of adata.
    """
    for col, labels in reference_labels.items():
        adata.obs[col] = labels.to_numpy()
    return adata


def rank_marker_genes(adata: sc.AnnData, groupby: str) -> sc.AnnData:
    """
    Ranks marker genes per group with a Wilcoxon test, reading log-normalized values
    from .raw rather than the scaled PCA-input matrix.
    """
    sc.tl.rank_genes_groups(adata, groupby, method="wilcoxon", use_raw=True)
    return adata


def export_marker_genes_md(
    adata: sc.AnnData,
    groupby: str,
    group_label: str,
    n_top_genes: int,
    title: str,
    instruction: str,
    out_path: str,
) -> str:
    """
    Exports the top marker genes per group to a markdown file for review.
    """
    marker_df = sc.get.rank_genes_groups_df(adata, group=None, key="rank_genes_groups")
    groups = sorted(adata.obs[groupby].unique(), key=lambda x: int(x))

    lines = [f"# {title}", "", instruction, ""]
    for group in groups:
        top_genes = (
            marker_df[marker_df["group"] == group]
            .nlargest(n_top_genes, "scores")["names"]
            .tolist()
        )
        lines.append(f"## {group_label} {group}")
        lines.append(", ".join(top_genes))
        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Exported marker genes for {len(groups)} groups to {out_path}")
    return out_path


def reduce_dims(
    adata: sc.AnnData,
    min_mean: float,
    max_mean: float,
    min_disp: float,
    n_comps: int,
) -> sc.AnnData:
    """
    Reduces to highly variable genes and runs PCA.
    """
    adata = select_hvgs(adata, min_mean, max_mean, min_disp)
    adata = run_pca(adata, n_comps)
    return adata


def cluster_cells(
    adata: sc.AnnData,
    n_pcs: int,
    n_neighbors: int,
    resolution: float,
    key_added: str,
) -> sc.AnnData:
    """
    Builds the neighborhood graph, Leiden-clusters, and computes the UMAP embedding.
    """
    adata = build_neighbors(adata, n_pcs, n_neighbors)
    adata = cluster_leiden(adata, resolution, key_added=key_added)
    adata = compute_umap(adata)
    return adata


def subset_cells(
    adata_full: sc.AnnData,
    adata_global: sc.AnnData,
    cluster_ids: list[str],
) -> sc.AnnData:
    """
    Subsets the full data to the cells in the given global Leiden clusters, carrying
    over the global cluster labels and reference annotations.
    """
    cell_mask = adata_global.obs["leiden"].isin(cluster_ids)
    cell_barcodes = adata_global.obs_names[cell_mask]
    adata_sub = adata_full[cell_barcodes].copy()
    for col in OBS_CARRY_COLS:
        if col in adata_global.obs.columns:
            adata_sub.obs[col] = adata_global.obs.loc[cell_barcodes, col]
    return adata_sub


def apply_cluster_annotations(
    adata: sc.AnnData,
    annotations: dict[str, str | None],
    groupby: str = "leiden",
    col: str = "CellType1",
) -> sc.AnnData:
    """
    Maps global cluster labels to annotations; unmapped cells become "Other".
    """
    adata.obs[col] = adata.obs[groupby].map(annotations).fillna("Other")
    return adata


def apply_subcluster_annotations(
    adata_global: sc.AnnData,
    adata_subs: dict[str, sc.AnnData],
    subcluster_annotations: dict[str, dict[str, str | None]],
    col: str = "CellType2",
) -> sc.AnnData:
    """
    Propagates sub-cluster annotations back onto the global object. Independent of the
    global annotations: cells without a sub-cluster label are "Other".
    """
    adata_global.obs[col] = "Other"
    for broad_label, sub_map in subcluster_annotations.items():
        if broad_label not in adata_subs:
            continue
        adata_sub = adata_subs[broad_label]
        for barcode, sub_id in adata_sub.obs["leiden_sub"].items():
            label = sub_map.get(sub_id)
            if label is not None:
                adata_global.obs.at[barcode, col] = label
    return adata_global


def export_annotations(
    adata: sc.AnnData, cols: list[str], out_path: str
) -> pd.DataFrame:
    """
    Exports the named annotation columns to a barcode-indexed CSV.
    """
    annot_df = adata.obs[cols].copy()
    annot_df.index.name = "barcode"
    annot_df.to_csv(out_path)
    print(f"Saved annotations for {len(annot_df)} cells to {out_path}")
    return annot_df
