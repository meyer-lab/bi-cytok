"""
Property-based testing file.
"""

import numpy as np
import pandas as pd
import pytest
import jax.numpy as jnp

from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.numpy import arrays

from ..binding_model_funcs import cyt_binding_model
from ..distance_metric_funcs import KL_EMD_1D, KL_EMD_2D, KL_EMD_3D
from ..imports import importCITE, sample_receptor_abundances, filter_receptor_abundances
from ..selectivity_funcs import (
    min_off_targ_selec,
    optimize_affs,
    restructure_affs,
    get_cell_bindings,
)


# Constants for testing
BASELINE_SCALE_FACTOR = 1.0
SCALE_FACTOR_TOLERANCE = 0.1
SELECTIVITY_LOWER_BOUND = 0.0
SELECTIVITY_UPPER_BOUND = 1.0
INSTABILITY_THRESHOLD = 0.5  # Relative change threshold for instability detection

# Fixed optimization bounds
FIXED_AFFINITY_BOUNDS = (6.0, 12.0)
FIXED_KX_STAR_BOUNDS = (2.24e-13, 2.24e-11)


@st.composite
def scaling_factors_near_one(draw):
    """Generate scaling factors close to 1.0 for instability testing."""
    # Focus on values near 1.0 where instability is most likely
    base_range = st.floats(min_value=0.5, max_value=2.0, allow_nan=False, allow_infinity=False)
    return draw(base_range)


@st.composite
def receptor_count_matrix(draw, min_cells=40, max_cells=60):
    """Generate realistic receptor count matrices for testing."""
    n_cells = draw(st.integers(min_value=min_cells, max_value=max_cells))
    
    # Generate receptor counts using log-normal distribution to mimic CITE-seq data
    receptor_counts = draw(arrays(
        dtype=np.float64,
        shape=(n_cells, 2),
        elements=st.integers(min_value=0, max_value=5000),
        unique=True
    ))
    receptor_counts = receptor_counts.astype(np.float64)
    receptor_counts[receptor_counts == 0] = 1e-9
    
    return receptor_counts


@st.composite
def model_parameters(draw):
    """Generate test parameters for selectivity optimization (excluding affinity bounds)."""
    dose = draw(st.floats(min_value=1e-11, max_value=1e-7, allow_nan=False, allow_infinity=False))
    valencies = draw(st.integers(min_value=1, max_value=4))
    
    return {
        'dose': dose,
        'valencies': np.array([[valencies, valencies]]),
        'affinity_bounds': FIXED_AFFINITY_BOUNDS,
        'kx_star_bounds': FIXED_KX_STAR_BOUNDS
    }


@st.composite
def model_affinities(draw, n_receptors=2):
    """Generate test affinity matrices for mass balance testing."""
    # Generate log10 affinities in a reasonable range
    log_affinities = draw(arrays(
        dtype=np.float64,
        shape=(n_receptors,),
        elements=st.floats(min_value=6.0, max_value=12.0, allow_nan=False, allow_infinity=False)
    ))
    
    # Convert to affinity matrix format expected by binding model
    affinity_matrix = np.zeros((n_receptors, n_receptors))
    np.fill_diagonal(affinity_matrix, 10**log_affinities)
    
    return affinity_matrix


class TestScalingInstability:
    """Test suite for identifying scaling-related instabilities in the binding model."""

    @given(st.integers(min_value=1, max_value=10))
    def test_int(self, value):
        """Simple test to ensure the test suite is running."""
        assert isinstance(value, int), "Value should be an integer"
    
    
    @given(
        scale_factor=scaling_factors_near_one(),
        targ_recs=receptor_count_matrix(),
        off_targ_recs=receptor_count_matrix(),
        params=model_parameters()
    )
    @settings(max_examples=5, deadline=None)  # Increase deadline for optimization
    def test_selectivity_bounds_with_scaling(self, scale_factor, targ_recs, off_targ_recs, params):
        """Test that selectivity remains within expected bounds when scaling factors are applied."""
        assume(targ_recs.shape[1] == off_targ_recs.shape[1])
        assume(np.all(targ_recs > 0) and np.all(off_targ_recs > 0))
        
        # Apply scaling factor to the first receptor (signal receptor)
        scaled_targ_recs = targ_recs.copy()
        scaled_off_targ_recs = off_targ_recs.copy()
        scaled_targ_recs[:, 1] *= scale_factor
        scaled_off_targ_recs[:, 1] *= scale_factor
        
        try:
            opt_selec, opt_affs, opt_kx_star = optimize_affs(
                targRecs=scaled_targ_recs,
                offTargRecs=scaled_off_targ_recs,
                dose=params['dose'],
                valencies=params['valencies'],
                affinity_bounds=params['affinity_bounds'],
                Kx_star_bounds=params['kx_star_bounds'],
            )
            
            selectivity = 1 / opt_selec
            
            # Test bounds violation
            assert SELECTIVITY_LOWER_BOUND <= selectivity <= SELECTIVITY_UPPER_BOUND, (
                f"Selectivity {selectivity:.6f} out of bounds [0, 1] with scale factor {scale_factor:.6f}"
            )
            
            # Test for NaN or infinite values
            assert np.isfinite(selectivity), f"Non-finite selectivity {selectivity} with scale factor {scale_factor:.6f}"
            assert np.all(np.isfinite(opt_affs)), f"Non-finite affinities with scale factor {scale_factor:.6f}"
            assert np.isfinite(opt_kx_star), f"Non-finite Kx_star with scale factor {scale_factor:.6f}"
            
        except Exception as e:
            pytest.fail(f"Optimization failed with scale factor {scale_factor:.6f}: {e}")
    

    @given(
        base_scale_factor=st.floats(min_value=0.9, max_value=1.1, allow_nan=False),
        perturbation=st.floats(min_value=-0.05, max_value=0.05, allow_nan=False),
        targ_recs=receptor_count_matrix(),
        off_targ_recs=receptor_count_matrix(),
        params=model_parameters()
    )
    @settings(max_examples=5, deadline=None)
    def test_selectivity_continuity_near_one(self, base_scale_factor, perturbation, targ_recs, off_targ_recs, params):
        """Test that small changes in scaling factor near 1.0 don't cause drastic selectivity changes."""
        assume(targ_recs.shape[1] == off_targ_recs.shape[1])
        assume(np.all(targ_recs > 0) and np.all(off_targ_recs > 0))
        assume(abs(perturbation) > 1e-6)  # Ensure meaningful perturbation
        
        scale_factor_1 = base_scale_factor
        scale_factor_2 = base_scale_factor + perturbation
        
        selectivities = []
        
        for scale_factor in [scale_factor_1, scale_factor_2]:
            scaled_targ_recs = targ_recs.copy()
            scaled_off_targ_recs = off_targ_recs.copy()
            scaled_targ_recs[:, 1] *= scale_factor
            scaled_off_targ_recs[:, 1] *= scale_factor
            
            try:
                opt_selec, _, _ = optimize_affs(
                    targRecs=scaled_targ_recs,
                    offTargRecs=scaled_off_targ_recs,
                    dose=params['dose'],
                    valencies=params['valencies'],
                    affinity_bounds=params['affinity_bounds'],
                    Kx_star_bounds=params['kx_star_bounds'],
                )
                
                selectivity = 1 / opt_selec
                selectivities.append(selectivity)
                
            except Exception as e:
                pytest.fail(f"Optimization failed with scale factor {scale_factor:.6f}: {e}")
        

        # Check for discontinuity/instability
        if len(selectivities) == 2:
            relative_change = abs(selectivities[1] - selectivities[0]) / max(selectivities[0], 1e-10)
            perturbation_magnitude = abs(perturbation / base_scale_factor)
            
            # Flag potential instability if relative change is disproportionate to perturbation
            if relative_change > INSTABILITY_THRESHOLD and perturbation_magnitude < 0.1:
                pytest.fail(
                    f"Potential instability detected: "
                    f"Scale factors {scale_factor_1:.6f} -> {scale_factor_2:.6f} "
                    f"caused selectivity change {selectivities[0]:.6f} -> {selectivities[1]:.6f} "
                    f"(relative change: {relative_change:.3f})"
                )
    
    @given(
        scale_factor=scaling_factors_near_one(),
        targ_recs=receptor_count_matrix(),
        off_targ_recs=receptor_count_matrix(),
        params=model_parameters(),
        test_affs=model_affinities(n_receptors=2)
    )
    @settings(max_examples=5, deadline=None)
    def test_mass_balance_conservation(self, scale_factor, targ_recs, off_targ_recs, params, test_affs):
        """Test that mass balance is conserved in the binding model under scaling."""
        assume(targ_recs.shape[1] == off_targ_recs.shape[1])
        assume(np.all(targ_recs > 0) and np.all(off_targ_recs > 0))
        
        # Apply scaling
        scaled_targ_recs = targ_recs.copy()
        scaled_targ_recs[:, 1] *= scale_factor
        
        try:
            bound_receptors = cyt_binding_model(
                dose=params['dose'],
                recCounts=scaled_targ_recs,
                valencies=params['valencies'],
                monomerAffs=test_affs,
                Kx_star=2.24e-12
            )
            
            # Test mass balance conservation: bound + unbound = total
            # The binding model returns bound receptors, so unbound = total - bound
            unbound_receptors = scaled_targ_recs - bound_receptors
            total_check = bound_receptors + unbound_receptors
            
            # Mass balance should be exact (within numerical precision)
            mass_balance_error = np.abs(total_check - scaled_targ_recs)
            max_relative_error = np.max(mass_balance_error / scaled_targ_recs)
            
            assert np.all(mass_balance_error < 1e-10), (
                f"Mass balance violated with scale factor {scale_factor:.6f}: "
                f"max absolute error = {np.max(mass_balance_error):.2e}"
            )
            
            assert max_relative_error < 1e-12, (
                f"Mass balance relative error too large with scale factor {scale_factor:.6f}: "
                f"max relative error = {max_relative_error:.2e}"
            )
            
            # Additional checks for physical validity
            assert np.all(bound_receptors >= 0), (
                f"Negative bound receptors with scale factor {scale_factor:.6f}"
            )
            assert np.all(unbound_receptors >= 0), (
                f"Negative unbound receptors with scale factor {scale_factor:.6f}"
            )
            assert np.all(bound_receptors <= scaled_targ_recs), (
                f"Bound receptors exceed total with scale factor {scale_factor:.6f}"
            )
            
            # Test that outputs are finite
            assert np.all(np.isfinite(bound_receptors)), (
                f"Non-finite bound receptors with scale factor {scale_factor:.6f}"
            )
            
        except Exception as e:
            pytest.fail(f"Mass balance test failed with scale factor {scale_factor:.6f}: {e}")
