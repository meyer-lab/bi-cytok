"""File that deals with everything about importing and sampling."""

import gzip
from pathlib import Path
from zipfile import ZipFile

import anndata as an
import numpy as np
import pandas as pd
from scipy.io import mmread

path_here = Path(__file__).parent.parent

SC_Stims = [
    "control",
    "IL2_100pM",
    "IL2_1nM",
    "IL2_10nM",
    "IL2_50nM",
    "IL2_200nM",
    "IL7_100nM",
    "IL10_500nM",
    "IL10_2000nM",
    "TGFB_10nM",
    "TGFB_50nM",
]  # "IL7_500nM is blank"


# Originally called in selectivityFuncs.getConvFactDict
def getBindDict():
    """Gets binding to pSTAT fluorescent conversion dictionary"""
    bindingDF = pd.read_csv(
        path_here / "bicytok" / "data" / "BindingConvDict.csv", encoding="latin1"
    )
    return bindingDF


# Sam: Not called anywhere, not sure what original use was
def importReceptors():
    """Makes Complete receptor expression Dict"""
    recDF = pd.read_csv(path_here / "bicytok" / "data" / "RecQuantitation.csv")
    recDFbin = pd.read_csv(path_here / "bicytok" / "data" / "BinnedReceptorData.csv")
    recDFbin = recDFbin.loc[recDFbin["Bin"].isin([1, 3])]
    recDFbin.loc[recDFbin["Bin"] == 1, "Cell Type"] += r" $IL2Ra^{lo}$"
    recDFbin.loc[recDFbin["Bin"] == 3, "Cell Type"] += r" $IL2Ra^{hi}$"
    recDF = pd.concat([recDF, recDFbin])
    return recDF


# Not called anywhere
def makeCITEdf():
    """Makes cite surface epitope csv for given cell type,
    DON'T USE THIS UNLESS DATA NEEDS RESTRUCTURING"""
    """
    matrixDF = pd.read_csv(join(path_here, "bicytok/data/CITEmatrix.gz"),
        compression='gzip', header=0, sep=' ', quotechar='"', error_bad_lines=False)
    matrixDF = matrixDF.iloc[:, 0:-2]
    matrixDF.columns = ["Marker", "Cell", "Number"]
    matrixDF.to_csv(join(path_here, "bicytok/data/CITEmatrix.csv"), index=False)
    """
    featureDF = pd.read_csv(path_here / "bicytok" / "data" / "CITEfeatures.csv")
    matrixDF = pd.read_csv(path_here / "bicytok" / "data" / "CITEmatrix.csv").iloc[
        1::, :
    ]
    metaDF = pd.read_csv(path_here / "bicytok" / "data" / "metaData3P.csv")

    metaDF["cellNumber"] = metaDF.index + 1
    cellNums = metaDF.cellNumber.values
    cellT1 = metaDF["celltype.l1"].values
    cellT2 = metaDF["celltype.l2"].values
    cellT3 = metaDF["celltype.l3"].values
    cellTDict1 = {cellNums[i]: cellT1[i] for i in range(len(cellNums))}
    cellTDict2 = {cellNums[i]: cellT2[i] for i in range(len(cellNums))}
    cellTDict3 = {cellNums[i]: cellT3[i] for i in range(len(cellNums))}

    featureDF["featNumber"] = featureDF.index + 1
    featNums = featureDF.featNumber.values
    features = featureDF.Marker.values
    featDict = {featNums[i]: features[i] for i in range(len(featNums))}
    matrixDF["Marker"] = matrixDF["Marker"].replace(featDict)

    categories1 = metaDF["celltype.l1"].unique()
    categories2 = metaDF["celltype.l2"].unique()
    categories3 = metaDF["celltype.l3"].unique()

    matrixDF = (
        matrixDF.pivot(index=["Cell"], columns="Marker", values="Number")
        .reset_index()
        .fillna(0)
    )

    matrixDF["CellType1"] = pd.Categorical(
        matrixDF["Cell"].replace(cellTDict1), categories=categories1
    )
    matrixDF["CellType2"] = pd.Categorical(
        matrixDF["Cell"].replace(cellTDict2), categories=categories2
    )
    matrixDF["CellType3"] = pd.Categorical(
        matrixDF["Cell"].replace(cellTDict3), categories=categories3
    )
    matrixDF.to_csv(path_here / "bicytok" / "data" / "CITEdata.csv", index=False)
    return matrixDF  # , featureDF, metaDF


def importCITE():
    """Downloads all surface markers and cell types"""
    CITEmarkerDF = pd.read_csv(
        path_here / "bicytok" / "data" / "CITEdata_SurfMarkers.zip"
    )
    return CITEmarkerDF


def importRNACITE():
    """Downloads all surface markers and cell types"""
    RNAsurfDF = pd.read_csv(
        ZipFile(path_here / "bicytok" / "data" / "RNAseqSurface.csv.zip").open(
            "RNAseqSurface.csv"
        )
    )
    return RNAsurfDF


# Sam: function not called anywhere, purpose unclear
def makeTregSC():
    """Constructs .h5ad file for PBMC stimulation experiment"""
    Treg_h5ad = an.AnnData()
    for i, stim in enumerate(SC_Stims):
        stim_an = an.AnnData()
        barcodes = pd.read_csv(
            gzip.open(
                path_here
                / "multi_output"
                / "outs"
                / "per_sample_outs"
                / stim
                / "count"
                / "sample_filtered_feature_bc_matrix"
                / "barcodes.tsv.gz"
            ),
            sep="\t",
            header=None,
        )
        matrix = mmread(
            gzip.open(
                path_here
                / "multi_output"
                / "outs"
                / "per_sample_outs"
                / stim
                / "count"
                / "sample_filtered_feature_bc_matrix"
                / "matrix.mtx.gz"
            )
        )
        barcodes.columns = ["barcode"]
        stim_an = an.AnnData(matrix.transpose())
        stim_an.obs.index = barcodes["barcode"].values
        stim_an.obs["Condition"] = stim

        if i == 0:  # First condition - load features for later labeling
            # (all conditions have same genes)
            Treg_h5ad = stim_an
            features = pd.read_csv(
                gzip.open(
                    path_here
                    / "multi_output"
                    / "outs"
                    / "per_sample_outs"
                    / stim
                    / "count"
                    / "sample_filtered_feature_bc_matrix"
                    / "features.tsv.gz"
                ),
                sep="\t",
                header=None,
            )
            features.columns = ["ENSEMBLE_ids", "gene_ids", "feature_type"]
        else:
            Treg_h5ad = an.concat([Treg_h5ad, stim_an])

    Treg_h5ad.var.index = features["gene_ids"].values
    Treg_h5ad.var["ENSEMBLE_ids"] = features["ENSEMBLE_ids"].values
    Treg_h5ad.var["feature_type"] = features["feature_type"].values

    Treg_h5ad.write_h5ad(path_here / "Treg_h5ads" / "Treg_raw.h5ad")

    return


# Sam: reimplement this function when we have a clearer idea
#   of how to calculate conversion factors
def calc_conv_facts() -> tuple[dict, float]:
    """
    Returns conversion factors by marker for converting CITEseq signal into abundance
    """

    # cellTypes = [
    #     "CD4 TCM",
    #     "CD8 Naive",
    #     "NK",
    #     "CD8 TEM",
    #     "CD4 Naive",
    #     "CD4 CTL",
    #     "CD8 TCM",
    #     "Treg",
    #     "CD4 TEM",
    # ]
    # markers = ["CD122", "CD127", "CD25"]
    # markDict = {
    #     "CD25": "IL2Ra",
    #     "CD122": "IL2Rb",
    #     "CD127": "IL7Ra",
    #     "CD132": "gc"
    # }
    # cellDict = {
    #     "CD4 Naive": "Thelper",
    #     "CD4 CTL": "Thelper",
    #     "CD4 TCM": "Thelper",
    #     "CD4 TEM": "Thelper",
    #     "NK": "NK",
    #     "CD8 Naive": "CD8",
    #     "CD8 TCM": "CD8",
    #     "CD8 TEM": "CD8",
    #     "Treg": "Treg",
    # }

    # Sam: calculation of these conversion factors was unclear, should be revised
    origConvFactDict = {
        "CD25": 77.136987,
        "CD122": 332.680090,
        "CD127": 594.379215,
    }
    convFactDict = origConvFactDict.copy()
    defaultConvFact = np.mean(list(origConvFactDict.values()))

    return convFactDict, defaultConvFact


def sample_receptor_abundances(
    CITE_DF: pd.DataFrame,
    numCells: int,
    targCellType: str,
    offTargCellTypes: list[str] = None,
    rand_state: int = 42,
    balance: bool = False,
    convert: bool = True,
) -> pd.DataFrame:
    """
    Samples a subset of cells and converts unprocessed CITE-seq receptor values
        into abundance values. Samples an equal number of target and off target cells.
    Args:
        CITE_DF: dataframe of unprocessed CITE-seq receptor counts
            of different receptors/epitopes (columns) on single cells (row).
            Epitopes are filtered outside of this function.
            The final column should be the cell types of each cell.
        numCells: number of cells to sample
        targCellType: the cell type that will be used to split target and
            off targer sampling
        offTargCellTypes: list of cell types that are distinct from target cells.
            If None, all cell types except targCellType will be used.
        rand_state: random seed for reproducibility
        balance: if True, forces sampling of an equal number of target and off-target
            cells
        convert: if True, converts CITE-seq signal into abundance values
            using conversion factors
    Return:
        sampleDF: dataframe containing single cell abundances of
            receptors (column) for each individual cell (row).
            The final column is the cell type of each cell.
    """

    assert numCells <= CITE_DF.shape[0]
    assert "Cell Type" in CITE_DF.columns

    # Sample an equal number of target and off-target cells
    target_cells = CITE_DF[CITE_DF["Cell Type"] == targCellType]
    if offTargCellTypes is not None:
        off_target_cells = CITE_DF[CITE_DF["Cell Type"].isin(offTargCellTypes)]
    else:
        off_target_cells = CITE_DF[CITE_DF["Cell Type"] != targCellType]

    # Split sample size between target and off-target cells. If insufficient target
    #   cells, fill the rest with off-target cells
    num_target_cells = min(numCells // 2, target_cells.shape[0])
    num_off_target_cells = min(numCells - num_target_cells, off_target_cells.shape[0])

    if balance:
        balanced_cell_count = min(num_target_cells, num_off_target_cells)
        num_target_cells = balanced_cell_count
        num_off_target_cells = balanced_cell_count

    sampled_target_cells = target_cells.sample(
        num_target_cells, random_state=rand_state
    )
    sampled_off_target_cells = off_target_cells.sample(
        num_off_target_cells, random_state=rand_state
    )

    sampleDF = pd.concat([sampled_target_cells, sampled_off_target_cells])

    # Calculate conversion factors for each epitope
    convFactDict, defaultConvFact = calc_conv_facts()

    # Multiply the receptor counts of epitope by the conversion factor for that epitope
    if convert:
        epitopes = CITE_DF.columns[CITE_DF.columns != "Cell Type"]
        convFacts = [convFactDict.get(epitope, defaultConvFact) for epitope in epitopes]
        sampleDF[epitopes] = sampleDF[epitopes] * convFacts

    return sampleDF


def filter_receptor_abundances(
    abundance_df: pd.DataFrame,
    targ_cell_type: str,
    min_mean_abundance: float = 5.0,
    epitope_list: list[str] = None,
    cell_type_list: list[str] = None,
) -> pd.DataFrame:
    """
    Filters receptor abundances by removing biologically irrelevant receptors and
        user specified epitopes and cell types. Biologically irrelevant receptors are
        defined as those with large enough mean abundance (can't target a receptor
        with low overall expression) and those that have higher expression in target
        cells compared to other cell types.
    Args:
        abundance_df: DataFrame containing receptor abundances for filtering
        targ_cell_type: The cell type to determine biologically relevant receptors
        min_mean_abundance: Minimum mean abundance threshold for receptors
        epitope_list: List of specific epitopes to retain; if None, all are retained
        cell_type_list: List of specific cell types to retain; if None, all are retained
    Return:
        A DataFrame containing filtered receptor abundances
    """

    assert "Cell Type" in abundance_df.columns

    cell_type_df = abundance_df["Cell Type"]
    abundance_df = abundance_df.drop(columns=["Cell Type"])

    # Filter irrelevant receptors
    mean_abundances = abundance_df.mean(axis=0)
    relevant_receptors = mean_abundances[mean_abundances > min_mean_abundance].index
    abundance_df = abundance_df[relevant_receptors]
    mean_targ_abundances = abundance_df[cell_type_df == targ_cell_type].mean(axis=0)
    mean_off_targ_abundances = abundance_df[cell_type_df != targ_cell_type].mean(axis=0)
    relevant_receptors = mean_targ_abundances[
        mean_targ_abundances > mean_off_targ_abundances
    ].index
    abundance_df = abundance_df[relevant_receptors]

    # Filter user-specified epitopes and cell types
    if epitope_list is not None:
        abundance_df = abundance_df[epitope_list]
    if cell_type_list is not None:
        abundance_df = abundance_df[cell_type_df.isin(cell_type_list)]
        cell_type_df = cell_type_df[cell_type_df.isin(cell_type_list)]

    # Re-add the cell type column efficiently using pd.concat
    epitope_cols = abundance_df.copy()
    cell_type_df = pd.DataFrame(cell_type_df, columns=["Cell Type"])
    abundance_df = pd.concat([epitope_cols, cell_type_df], axis=1)

    return abundance_df
