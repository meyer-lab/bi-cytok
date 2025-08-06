"""
Property-based testing file.
"""

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from ..selectivity_funcs import get_cell_bindings, optimize_affs

# Constants for testing
instability_threshold = 0.01  # Relative change threshold for instability detection
num_output_tests = 50
num_scaling_tests = 50

# Model constants
affinity_bounds = (6.0, 12.0)
Kx_star_bounds = (2.24e-15, 2.24e-9)
dose_bounds = (1e-13, 1e-7)
valency_bounds = (1, 2)
cell_count = 50


@st.composite
def receptor_count_matrix(draw):
    """Generate receptor count matrices for testing."""

    # Generate receptor counts using log-normal distribution to mimic CITE-seq data
    receptor_counts = draw(
        arrays(
            dtype=np.float64,
            shape=(cell_count, 2),
            elements=st.integers(min_value=0, max_value=5000),
            unique=True,
        )
    )
    receptor_counts = receptor_counts.astype(np.float64)
    receptor_counts[receptor_counts == 0] = 1e-9

    return receptor_counts


@st.composite
def model_parameters(draw):
    """Generate dose and valency parameters for selectivity optimization."""
    dose = draw(
        st.floats(
            min_value=dose_bounds[0],
            max_value=dose_bounds[1],
            allow_nan=False,
            allow_infinity=False,
        )
    )
    valencies = draw(
        st.integers(min_value=valency_bounds[0], max_value=valency_bounds[1])
    )

    return {
        "dose": dose,
        "valencies": np.array([[valencies, valencies]]),
    }


@st.composite
def scaling_factors_near_one(draw):
    """Generate scaling factors close to 1.0 for instability testing."""
    base_range = st.floats(
        min_value=0.5, max_value=2.0, allow_nan=False, allow_infinity=False
    )
    return draw(base_range)


class TestModelInstability:
    """Test suite for identifying scaling-related instabilities in the binding model."""

    @given(
        targ_recs=receptor_count_matrix(),
        off_targ_recs=receptor_count_matrix(),
        params=model_parameters(),
    )
    @settings(max_examples=num_output_tests, deadline=None)
    def test_model_outputs(self, targ_recs, off_targ_recs, params):
        """Test that the model returns logical values when changing receptor counts, valencies, and doses."""
        assume(np.all(targ_recs > 0) and np.all(off_targ_recs > 0))

        try:
            opt_selec, opt_affs, opt_kx_star = optimize_affs(
                targRecs=targ_recs,
                offTargRecs=off_targ_recs,
                dose=params["dose"],
                valencies=params["valencies"],
                affinity_bounds=affinity_bounds,
                Kx_star_bounds=Kx_star_bounds,
            )

            selectivity = 1 / opt_selec

            Rbound = get_cell_bindings(
                dose=params["dose"],
                recCounts=targ_recs,
                valencies=params["valencies"],
                monomerAffs=opt_affs,
                Kx_star=opt_kx_star,
            )

            # Test Rbound mass balance
            assert np.all(Rbound <= targ_recs), "Bound receptors exceed total receptors"

            # Test selectivity bounds violation
            assert 0 <= selectivity <= 1, (
                f"Selectivity {selectivity:.6f} out of bounds [0, 1]"
            )

            # Test for NaN or infinite values
            assert np.isfinite(selectivity), f"Non-finite selectivity {selectivity}"
            assert np.all(np.isfinite(opt_affs)), f"Non-finite affinities {opt_affs}"
            assert np.isfinite(opt_kx_star), f"Non-finite Kx_star {opt_kx_star}"

        except Exception as e:
            pytest.fail(f"Optimization failed with error: {e}")

    @given(
        base_scale_factor=st.floats(min_value=0.9, max_value=1.1, allow_nan=False),
        perturbation=st.floats(min_value=-0.05, max_value=0.05, allow_nan=False),
        targ_recs=receptor_count_matrix(),
        off_targ_recs=receptor_count_matrix(),
        params=model_parameters(),
    )
    @settings(max_examples=num_scaling_tests, deadline=None)
    def test_selectivity_continuity_near_one(
        self, base_scale_factor, perturbation, targ_recs, off_targ_recs, params
    ):
        """Test that small changes in scaling factor near 1.0 don't cause drastic selectivity changes."""
        assume(np.all(targ_recs > 0) and np.all(off_targ_recs > 0))

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
                    dose=params["dose"],
                    valencies=params["valencies"],
                    affinity_bounds=affinity_bounds,
                    Kx_star_bounds=Kx_star_bounds,
                )

                selectivity = 1 / opt_selec
                selectivities.append(selectivity)

            except Exception as e:
                pytest.fail(f"Optimization failed with error: {e}")

        # Check for discontinuity/instability
        relative_change = abs(selectivities[1] - selectivities[0]) / max(
            selectivities[0], 1e-10
        )

        # Flag potential instability if relative change is large
        if relative_change > instability_threshold:
            pytest.fail(
                f"Potential instability detected: "
                f"Scale factors {scale_factor_1:.6f} -> {scale_factor_2:.6f} "
                f"caused selectivity change {selectivities[0]:.6f} -> {selectivities[1]:.6f} "
                f"(relative change: {relative_change:.12f})"
            )
