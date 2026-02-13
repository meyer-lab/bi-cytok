"""File for importing and sampling CITE-seq data."""

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

path_here = Path(__file__).parent.parent


def importCITE():
    """Downloads all surface markers and cell types"""
    CITEmarkerDF = pd.read_csv(path_here / "data" / "CITEdata_SurfMarkers.zip")
    return CITEmarkerDF


def importRNACITE():
    """Downloads all surface markers and cell types"""
    with ZipFile(path_here / "data" / "RNAseqSurface.csv.zip") as zip_file:
        RNAsurfDF = pd.read_csv(zip_file.open("RNAseqSurface.csv"))
    return RNAsurfDF


def sample_receptor_abundances(
    CITE_DF: pd.DataFrame,
    numCells: int,
    targCellType: str,
    offTargCellTypes: list[str] = None,
    rand_state: int = 42,
    balance: bool = False,
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

    return sampleDF


def filter_receptor_abundances(
    abundance_df: pd.DataFrame,
    targ_cell_type: str,
    min_mean_abundance: float = 5.0,
    whitelist: list[str] = None,
    blacklist: list[str] = None,
) -> pd.DataFrame:
    """
    Filters receptor abundances by removing biologically irrelevant receptors.
        Biologically irrelevant receptors are defined as those with large enough mean
        abundance (can't target a receptor with low overall expression) and those that
        have higher expression in target cells compared to other cell types.
        Whitelisted receptors are always included and blacklisted receptors excluded.
    Args:
        abundance_df: DataFrame containing receptor abundances for filtering
        targ_cell_type: The cell type to determine biologically relevant receptors
        min_mean_abundance: Minimum mean abundance threshold for receptors
        whitelist: List of receptors to include regardless of filtering criteria
        blacklist: List of receptors to exclude regardless of filtering criteria
    Return:
        A DataFrame containing filtered receptor abundances
    """

    assert "Cell Type" in abundance_df.columns, "Missing cell type annotations"

    whitelist = whitelist or []
    blacklist = blacklist or []
    assert not [r for r in whitelist if r not in abundance_df.columns], (
        "Whitelist receptors not found in data"
    )
    assert not set(whitelist).intersection(set(blacklist)), (
        "Overlap between whitelisted and blacklisted receptors"
    )

    # Separate cell type column for filtering
    cell_type_df = abundance_df["Cell Type"]
    abundance_df = abundance_df.drop(columns=["Cell Type"])

    # Remove blacklisted receptors
    abundance_df = abundance_df.drop(columns=blacklist, errors="ignore")

    # Filter irrelevant receptors based on mean abundance
    mean_abundances = abundance_df.mean(axis=0)
    high_mean = list(mean_abundances[mean_abundances > min_mean_abundance].index)

    # Filter based on target vs off-target expression
    mean_targ_abundances = abundance_df[cell_type_df == targ_cell_type].mean(axis=0)
    mean_off_targ_abundances = abundance_df[cell_type_df != targ_cell_type].mean(axis=0)
    higher_in_target = list(
        mean_targ_abundances[mean_targ_abundances > mean_off_targ_abundances].index
    )

    # Apply filtering (receptors must pass both filters or be whitelisted)
    relevant_receptors = list(
        set(high_mean).intersection(set(higher_in_target)).union(set(whitelist))
    )
    abundance_df = abundance_df[relevant_receptors]

    # Re-add the cell type column
    abundance_df = pd.concat([abundance_df, cell_type_df], axis=1)

    return abundance_df


def sample_test_data(n_obs=100, n_var=10):
    """
    Creates synthetic receptor abundance data for binding model and distribution metric
       unit testing.

    Args:
        n_obs: number of observations (cells)
        n_var: number of variables (receptors)
    Returns:
        recAbundances: synthetic receptor abundance data
        targ: boolean array indicating target cells
        offTarg: boolean array indicating off-target cells
    """
    rng = np.random.default_rng(42)
    recAbundances = rng.normal(loc=50, scale=2, size=(n_obs, n_var))
    targ_ind = rng.choice(n_obs, size=n_obs // 2, replace=False)
    targ = np.zeros(n_obs, dtype=bool)
    targ[targ_ind] = True
    offTarg = ~targ

    # Scale target receptor abundances to higher mean
    recAbundances[targ] = recAbundances[targ] + rng.random() * 5.5 + 0.5

    return recAbundances, targ, offTarg
