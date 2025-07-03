"""
Generates plots comparing different selectivity definitions and their sensitivity
to scaling factors of receptor counts.

This figure explores how different ways of calculating selectivity
(signal-only, averaged, weighted) respond to changes in receptor abundance scaling.

Data Import:
- The CITE-seq dataframe (`importCITE`)

Parameters:
- targCell: cell type whose selectivity will be maximized
- sample_size: number of cells to sample from the CITE-seq dataframe
- cell_categorization: column name in the CITE-seq dataframe that categorizes cells
- model_valencies: valencies each receptor's ligand in the model molecule
- dose: dose of the model molecule
- signal_receptor: receptor that is the target of the model molecule
- target_receptors: list of receptors to be tested for selectivity
- affinity_bounds: optimization bounds for the affinities of the receptors

Outputs:
- A multi-panel plot comparing how different selectivity definitions respond to scaling
- Shows selectivity values, optimal affinities, and binding patterns across scaling factors
"""

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import Bounds, minimize

from ..imports import importCITE, sample_receptor_abundances
from ..selectivity_funcs import restructure_affs, optimize_affs, get_cell_bindings
from ..binding_model_funcs import cyt_binding_model
from .common import getSetup

path_here = Path(__file__).parent.parent


def min_off_targ_selec_signal_only(
    monomerAffs: np.ndarray,
    targRecs: np.ndarray,
    offTargRecs: np.ndarray,
    dose: float,
    valencies: np.ndarray,
) -> float:
    """Original selectivity definition - signal receptor only (column 0)."""
    assert targRecs.shape[1] == offTargRecs.shape[1]
    
    modelAffs = restructure_affs(monomerAffs)
    
    targRbound, _ = cyt_binding_model(
        dose=dose, recCounts=targRecs, valencies=valencies, monomerAffs=modelAffs
    )
    offTargRbound, _ = cyt_binding_model(
        dose=dose, recCounts=offTargRecs, valencies=valencies, monomerAffs=modelAffs
    )
    
    fullRbound = np.concatenate((targRbound, offTargRbound), axis=0)
    
    targetBound = np.sum(targRbound[:, 0]) / targRbound.shape[0]
    fullBound = np.sum(fullRbound[:, 0]) / fullRbound.shape[0]
    
    return fullBound / targetBound


def min_off_targ_selec_signal_only_off_target(
    monomerAffs: np.ndarray,
    targRecs: np.ndarray,
    offTargRecs: np.ndarray,
    dose: float,
    valencies: np.ndarray,
) -> float:
    """Signal-only selectivity using off-target/target ratio."""
    assert targRecs.shape[1] == offTargRecs.shape[1]
    
    modelAffs = restructure_affs(monomerAffs)
    
    targRbound, _ = cyt_binding_model(
        dose=dose, recCounts=targRecs, valencies=valencies, monomerAffs=modelAffs
    )
    offTargRbound, _ = cyt_binding_model(
        dose=dose, recCounts=offTargRecs, valencies=valencies, monomerAffs=modelAffs
    )
    
    targetBound = np.sum(targRbound[:, 0]) / targRbound.shape[0]
    offTargetBound = np.sum(offTargRbound[:, 0]) / offTargRbound.shape[0]
    
    return offTargetBound / targetBound


def min_off_targ_selec_weighted(
    monomerAffs: np.ndarray,
    targRecs: np.ndarray,
    offTargRecs: np.ndarray,
    dose: float,
    valencies: np.ndarray,
) -> float:
    """Weighted selectivity definition - weighted by receptor abundance."""
    assert targRecs.shape[1] == offTargRecs.shape[1]
    assert targRecs.shape[1] >= 2
    
    modelAffs = restructure_affs(monomerAffs)
    
    targRbound, _ = cyt_binding_model(
        dose=dose, recCounts=targRecs, valencies=valencies, monomerAffs=modelAffs
    )
    offTargRbound, _ = cyt_binding_model(
        dose=dose, recCounts=offTargRecs, valencies=valencies, monomerAffs=modelAffs
    )
    
    fullRbound = np.concatenate((targRbound, offTargRbound), axis=0)
    
    # Calculate weights based on average receptor abundance
    totalAbundance = np.mean(targRecs[:, 0]) + np.mean(targRecs[:, 1])
    signalWeight = np.mean(targRecs[:, 0]) / totalAbundance
    targetWeight = np.mean(targRecs[:, 1]) / totalAbundance
    
    # Calculate selectivity for each receptor
    targetBoundSignal = np.sum(targRbound[:, 0]) / targRbound.shape[0]
    fullBoundSignal = np.sum(fullRbound[:, 0]) / fullRbound.shape[0]
    signalSelectivity = fullBoundSignal / targetBoundSignal
    
    targetBoundTarget = np.sum(targRbound[:, 1]) / targRbound.shape[0]
    fullBoundTarget = np.sum(fullRbound[:, 1]) / fullRbound.shape[0]
    targetSelectivity = fullBoundTarget / targetBoundTarget
    
    return signalWeight * signalSelectivity + targetWeight * targetSelectivity


def min_off_targ_selec_weighted_off_target(
    monomerAffs: np.ndarray,
    targRecs: np.ndarray,
    offTargRecs: np.ndarray,
    dose: float,
    valencies: np.ndarray,
) -> float:
    """Weighted selectivity using off-target/target ratio - weighted by receptor abundance."""
    assert targRecs.shape[1] == offTargRecs.shape[1]
    assert targRecs.shape[1] >= 2
    
    modelAffs = restructure_affs(monomerAffs)
    
    targRbound, _ = cyt_binding_model(
        dose=dose, recCounts=targRecs, valencies=valencies, monomerAffs=modelAffs
    )
    offTargRbound, _ = cyt_binding_model(
        dose=dose, recCounts=offTargRecs, valencies=valencies, monomerAffs=modelAffs
    )
    
    # Calculate weights based on average receptor abundance
    totalAbundance = np.mean(targRecs[:, 0]) + np.mean(targRecs[:, 1])
    signalWeight = np.mean(targRecs[:, 0]) / totalAbundance
    targetWeight = np.mean(targRecs[:, 1]) / totalAbundance
    
    # Calculate selectivity for each receptor using off-target/target ratios
    targetBoundSignal = np.sum(targRbound[:, 0]) / targRbound.shape[0]
    offTargetBoundSignal = np.sum(offTargRbound[:, 0]) / offTargRbound.shape[0]
    signalSelectivity = offTargetBoundSignal / targetBoundSignal
    
    targetBoundTarget = np.sum(targRbound[:, 1]) / targRbound.shape[0]
    offTargetBoundTarget = np.sum(offTargRbound[:, 1]) / offTargRbound.shape[0]
    targetSelectivity = offTargetBoundTarget / targetBoundTarget
    
    return signalWeight * signalSelectivity + targetWeight * targetSelectivity


def optimize_affs_custom(
    targRecs: np.ndarray,
    offTargRecs: np.ndarray,
    dose: float,
    valencies: np.ndarray,
    selectivity_func,
    bounds: tuple[float, float] = (7.0, 9.0),
) -> tuple[float, np.ndarray]:
    """Custom optimization function with selectable selectivity definition."""
    assert targRecs.size > 0
    assert offTargRecs.size > 0
    
    minAffs = [bounds[0]] * (targRecs.shape[1])
    maxAffs = [bounds[1]] * (targRecs.shape[1])
    
    initAffs = np.full_like(valencies[0], minAffs[0] + (maxAffs[0] - minAffs[0]) / 2)
    optBnds = Bounds(np.full_like(initAffs, minAffs), np.full_like(initAffs, maxAffs))
    
    targRecs[targRecs == 0] = 1e-9
    offTargRecs[offTargRecs == 0] = 1e-9
    
    optimizer = minimize(
        fun=selectivity_func,
        x0=initAffs,
        bounds=optBnds,
        args=(targRecs, offTargRecs, dose, valencies),
        jac="3-point",
    )
    
    return optimizer.fun, optimizer.x


def makeFigure():
    ax, f = getSetup((18, 10), (2, 3))  # 2x3 layout for comparison plots

    # Parameters
    targCell = "Treg"
    sample_size = 100
    cell_categorization = "CellType2"
    model_valencies = np.array([[(2), (2)]])
    dose = 10e-2
    signal_receptor = "CD122"
    target_receptors = ["CD25"]  # Focus on one target for clarity
    affinity_bounds = (5, 15)
    num_conv_factors = 15

    CITE_DF = importCITE()

    epitopes_all = [
        col
        for col in CITE_DF.columns
        if col not in ["CellType1", "CellType2", "CellType3"]
    ]
    epitopesDF = CITE_DF[epitopes_all + [cell_categorization]]
    epitopesDF = epitopesDF.rename(columns={cell_categorization: "Cell Type"})

    sampleDF = sample_receptor_abundances(
        CITE_DF=epitopesDF,
        numCells=sample_size,
        targCellType=targCell,
        convert=False,
    )

    # Normalize target receptor to signal receptor
    signal_mean = np.mean(sampleDF[signal_receptor])
    for receptor in target_receptors:
        target_mean = np.mean(sampleDF[receptor])
        sampleDF[receptor] = sampleDF[receptor] * (signal_mean / target_mean)

    # Create filtered dataframe with signal and target receptors
    filterDF = pd.DataFrame(
        {
            signal_receptor: sampleDF[signal_receptor],
            target_receptors[0]: sampleDF[target_receptors[0]],
            "Cell Type": sampleDF["Cell Type"],
        }
    )

    on_target_mask = (filterDF["Cell Type"] == targCell).to_numpy()
    off_target_mask = ~on_target_mask

    # Define selectivity functions to compare
    selectivity_definitions = {
        'Signal Only (Full/Target)': min_off_targ_selec_signal_only,
        'Signal Only (Off/Target)': min_off_targ_selec_signal_only_off_target,
        'Weighted (Full/Target)': min_off_targ_selec_weighted,
        'Weighted (Off/Target)': min_off_targ_selec_weighted_off_target,
    }

    # Calculate selectivity for each conversion factor
    conversion_factors = np.logspace(-2, 8, num=num_conv_factors)
    
    results = {}
    for def_name, selec_func in selectivity_definitions.items():
        selectivities = []
        optimal_signal_affs = []
        optimal_target_affs = []
        
        for conv_fact in conversion_factors:
            test_DF = filterDF.copy()
            test_DF[signal_receptor] = test_DF[signal_receptor] * conv_fact
            test_DF[target_receptors[0]] = test_DF[target_receptors[0]] * conv_fact
            rec_mat = test_DF[[signal_receptor, target_receptors[0]]].to_numpy()

            try:
                optSelec, optAffs = optimize_affs_custom(
                    targRecs=rec_mat[on_target_mask],
                    offTargRecs=rec_mat[off_target_mask],
                    dose=dose,
                    valencies=model_valencies,
                    selectivity_func=selec_func,
                    bounds=affinity_bounds,
                )
                selectivities.append(1 / optSelec)  # Invert for better interpretation
                optimal_signal_affs.append(optAffs[0])
                optimal_target_affs.append(optAffs[1])
            except:
                selectivities.append(np.nan)
                optimal_signal_affs.append(np.nan)
                optimal_target_affs.append(np.nan)

        results[def_name] = {
            'selectivity': np.array(selectivities),
            'signal_affs': np.array(optimal_signal_affs),
            'target_affs': np.array(optimal_target_affs),
        }

    # Plotting
    palette = sns.color_palette("colorblind", n_colors=len(selectivity_definitions))
    colors = {name: palette[i] for i, name in enumerate(selectivity_definitions.keys())}

    # Plot 1: Selectivity comparison
    for def_name, data in results.items():
        ax[0].plot(
            conversion_factors,
            data['selectivity'],
            marker="o",
            ls="-",
            label=def_name,
            color=colors[def_name],
            linewidth=1.5,
            markersize=4,
        )
    ax[0].axvline(x=10**0, linestyle="--", color="black", alpha=0.5)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("Conversion Factor")
    ax[0].set_ylabel("Optimal Selectivity")
    ax[0].set_title("Selectivity vs Conversion Factor")
    ax[0].legend(title="Selectivity Definition")

    # Plot 2: Signal receptor optimal affinities
    for def_name, data in results.items():
        ax[1].plot(
            conversion_factors,
            data['signal_affs'],
            marker="o",
            ls="-",
            label=def_name,
            color=colors[def_name],
            linewidth=1.5,
            markersize=4,
        )
    ax[1].axvline(x=10**0, linestyle="--", color="black", alpha=0.5)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("Conversion Factor")
    ax[1].set_ylabel("Optimal Signal Affinity (log10 Ka)")
    ax[1].set_title(f"Signal Receptor ({signal_receptor}) Optimal Affinity")
    ax[1].legend(title="Selectivity Definition")

    # Plot 3: Target receptor optimal affinities
    for def_name, data in results.items():
        ax[2].plot(
            conversion_factors,
            data['target_affs'],
            marker="o",
            ls="-",
            label=def_name,
            color=colors[def_name],
            linewidth=1.5,
            markersize=4,
        )
    ax[2].axvline(x=10**0, linestyle="--", color="black", alpha=0.5)
    ax[2].set_xscale("log")
    ax[2].set_xlabel("Conversion Factor")
    ax[2].set_ylabel("Optimal Target Affinity (log10 Ka)")
    ax[2].set_title(f"Target Receptor ({target_receptors[0]}) Optimal Affinity")
    ax[2].legend(title="Selectivity Definition")

    # Plot 4: Selectivity sensitivity (rate of change)
    for def_name, data in results.items():
        selectivities = data['selectivity']
        valid_idx = ~np.isnan(selectivities)
        if np.sum(valid_idx) > 1:
            # Calculate relative change in selectivity
            log_selec = np.log10(selectivities[valid_idx])
            sensitivity = np.abs(np.diff(log_selec))
            cf_midpoints = conversion_factors[valid_idx][1:]
            
            ax[3].plot(
                cf_midpoints,
                sensitivity,
                marker="o",
                ls="-",
                label=def_name,
                color=colors[def_name],
                linewidth=1.5,
                markersize=4,
            )
    ax[3].axvline(x=10**0, linestyle="--", color="black", alpha=0.5)
    ax[3].set_xscale("log")
    ax[3].set_xlabel("Conversion Factor")
    ax[3].set_ylabel("Selectivity Sensitivity (|Δlog10|)")
    ax[3].set_title("Selectivity Sensitivity vs Conversion Factor")
    ax[3].legend(title="Selectivity Definition")

    # Plot 5: Affinity difference (Target - Signal)
    for def_name, data in results.items():
        aff_diff = data['target_affs'] - data['signal_affs']
        ax[4].plot(
            conversion_factors,
            aff_diff,
            marker="o",
            ls="-",
            label=def_name,
            color=colors[def_name],
            linewidth=1.5,
            markersize=4,
        )
    ax[4].axvline(x=10**0, linestyle="--", color="black", alpha=0.5)
    ax[4].set_xscale("log")
    ax[4].set_xlabel("Conversion Factor")
    ax[4].set_ylabel("Affinity Difference (Target - Signal)")
    ax[4].set_title("Receptor Affinity Difference vs Conversion Factor")
    ax[4].legend(title="Selectivity Definition")

    # Plot 6: Selectivity ratio comparison (normalized to signal-only at conversion factor = 1)
    baseline_idx = np.argmin(np.abs(conversion_factors - 1.0))
    signal_only_baseline = results['Signal Only (Full/Target)']['selectivity'][baseline_idx]
    
    for def_name, data in results.items():
        normalized_selec = data['selectivity'] / signal_only_baseline
        ax[5].plot(
            conversion_factors,
            normalized_selec,
            marker="o",
            ls="-",
            label=def_name,
            color=colors[def_name],
            linewidth=1.5,
            markersize=4,
        )
    ax[5].axvline(x=10**0, linestyle="--", color="black", alpha=0.5)
    ax[5].axhline(y=1.0, linestyle=":", color="gray", alpha=0.7)
    ax[5].set_xscale("log")
    ax[5].set_xlabel("Conversion Factor")
    ax[5].set_ylabel("Normalized Selectivity (vs Signal-Only Baseline)")
    ax[5].set_title("Relative Selectivity Performance")
    ax[5].legend(title="Selectivity Definition")

    return f
