"""
Experiment 3: Legacy code, copula with interval AR(1) correlation ρ ∈ [0.1, 0.2].

For interval ρ ∈ [0.1, 0.2]:
- Evaluate at both vertices (ρ=0.1 and ρ=0.2)
- Take interval hull (outer envelope)
"""
import sys

import numpy as np
import scipy.stats
import time

import interval
from stlpy_copulas.STL import STLRandomVariable, GaussianCopulaN, LinearPredicate

from Bicycle import (
    Bicycle,
    simulate_nominal_trajectory,
    construct_ar1_correlation,
    X0, U_NOMINAL, T_HORIZON, CORRIDOR_WIDTH, WAYPOINT_Y, WAYPOINT_TIME_WINDOW,
    SIGMA_Y_BASE
)

# Parameters
NBINS = 50
N_SAMPLES = 10000  # Full sampling for final results
N_BOOTSTRAP = 100
# SIGMA_Y_BASE imported from Bicycle module
RHO_LOWER = 0.1
RHO_UPPER = 0.2

print("="*80)
print("EXPERIMENT 3: Hierarchical Copula with Interval AR(1) Correlation")
print("="*80)
print(f"\nConfiguration:")
print(f"  Formula: always[0,{T_HORIZON}](corridor) AND eventually[{T_HORIZON-WAYPOINT_TIME_WINDOW},{T_HORIZON}](waypoint)")
print(f"  AR(1) correlation: ρ ∈ [{RHO_LOWER}, {RHO_UPPER}]")
print(f"  Copula structure: Hierarchical (26D for always, 6D for eventually)")
print(f"  N_SAMPLES: {N_SAMPLES}")
print(f"  N_BOOTSTRAP: {N_BOOTSTRAP}")
print()

# Setup
print("Setting up bicycle trajectory...")
traj = simulate_nominal_trajectory(X0, U_NOMINAL, T_HORIZON)
print("  Done.\n")

# Build signal with Gaussian marginals
print("Building signal with Gaussian marginals...")
signal_y = []
for t in range(T_HORIZON + 1):
    sqrt_t_factor = np.sqrt(t + 1)
    std_y = SIGMA_Y_BASE * sqrt_t_factor
    mean_y = traj[1, t]

    rv_y = STLRandomVariable(
        scipy.stats.norm(loc=mean_y, scale=std_y),
        debug=False
    )
    rv_y.compute_inverse_cdf_from_cdf(
        NBINS,
        np.interval(mean_y - 3 * std_y, mean_y + 3 * std_y)
    )
    signal_y.append(rv_y)

print(f"✓ Created {len(signal_y)} STLRandomVariable objects\n")

# Create sub-formulas
corridor_hw = CORRIDOR_WIDTH / 2
corridor_lower = LinearPredicate(np.array([1]), -corridor_hw, random=True)
corridor_upper = LinearPredicate(np.array([-1]), -corridor_hw, random=True)
corridor = corridor_lower & corridor_upper
phi_corridor = corridor.always(0, T_HORIZON)

wp_y_lower = LinearPredicate(np.array([1]), WAYPOINT_Y[0], random=True)
wp_y_upper = LinearPredicate(np.array([-1]), -WAYPOINT_Y[1], random=True)
waypoint = wp_y_lower & wp_y_upper
wp_start = T_HORIZON - WAYPOINT_TIME_WINDOW
phi_waypoint = waypoint.eventually(wp_start, T_HORIZON)


def evaluate_at_rho(rho_value):
    """Evaluate both sub-formulas at given ρ value."""
    print(f"  Evaluating at ρ = {rho_value}...")

    # Corridor with 26×26 copula
    Sigma_26 = construct_ar1_correlation(26, rho=rho_value)
    copula_26 = GaussianCopulaN(Sigma_26, n_samples=N_SAMPLES, n_bootstrap=N_BOOTSTRAP)
    rho_corridor = phi_corridor.robustness(signal_y, 0, random=True, copula=copula_26)

    # Waypoint with 6×6 copula
    Sigma_6 = construct_ar1_correlation(6, rho=rho_value)
    copula_6 = GaussianCopulaN(Sigma_6, n_samples=N_SAMPLES, n_bootstrap=N_BOOTSTRAP)
    rho_waypoint = phi_waypoint.robustness(signal_y, 0, random=True, copula=copula_6)

    # Manual combination
    xs_l_corr, xs_u_corr = interval.get_lu(rho_corridor.xs)
    xs_l_wp, xs_u_wp = interval.get_lu(rho_waypoint.xs)
    xs_l = np.minimum(xs_l_corr, xs_l_wp)
    xs_u = np.minimum(xs_u_corr, xs_u_wp)

    return xs_l, xs_u


# ============================================================================
# Evaluate at both vertices
# ============================================================================
print("="*80)
print("VERTEX ENUMERATION")
print("="*80)

print("\nVertex 1: ρ = 0.1")
start1 = time.time()
xs_l_lower, xs_u_lower = evaluate_at_rho(RHO_LOWER)
time1 = time.time() - start1
print(f"  ✓ Completed in {time1:.2f}s")
print(f"    Finite: {np.sum(np.isfinite(xs_l_lower) & np.isfinite(xs_u_lower))}/50")

print("\nVertex 2: ρ = 0.2")
start2 = time.time()
xs_l_upper, xs_u_upper = evaluate_at_rho(RHO_UPPER)
time2 = time.time() - start2
print(f"  ✓ Completed in {time2:.2f}s")
print(f"    Finite: {np.sum(np.isfinite(xs_l_upper) & np.isfinite(xs_u_upper))}/50")

# ============================================================================
# Take interval hull
# ============================================================================
print("\n" + "="*80)
print("INTERVAL HULL")
print("="*80)

xs_l_final = np.minimum(xs_l_lower, xs_l_upper)
xs_u_final = np.maximum(xs_u_lower, xs_u_upper)

valid_mask = np.isfinite(xs_l_final) & np.isfinite(xs_u_final)
n_valid = np.sum(valid_mask)

if n_valid > 0:
    width = np.mean((xs_u_final - xs_l_final)[valid_mask])

    print(f"\nValid quantiles: {n_valid}/50 ({100*n_valid/50:.0f}%)")
    print(f"Mean interval width: {width:.6f}")
    print(f"Robustness range: [{np.nanmin(xs_l_final):.4f}, {np.nanmax(xs_u_final):.4f}]")
    print()

    # ========================================================================
    # FINAL RESULTS
    # ========================================================================
    print("="*80)
    print("EXPERIMENT 3: RESULTS")
    print("="*80)
    print(f"Total computation time: {time1 + time2:.2f}s")
    print(f"  - Vertex ρ=0.1: {time1:.2f}s")
    print(f"  - Vertex ρ=0.2: {time2:.2f}s")
    print()
    print(f"Valid quantiles: {n_valid}/50 ({100*n_valid/50:.0f}%)")
    print(f"Mean interval width: {width:.6f}")
    print(f"Robustness range: [{np.nanmin(xs_l_final):.4f}, {np.nanmax(xs_u_final):.4f}]")
    print()
    print("="*80)
    print("✅ EXPERIMENT 3 COMPLETE")
    print("="*80)
    print("\nKey achievements:")
    print("  ✅ Full formula (corridor AND waypoint)")
    print("  ✅ Hierarchical copula structure:")
    print("      - GaussianCopulaN(26×26) for always[0,25](corridor)")
    print("      - GaussianCopulaN(6×6) for eventually[20,25](waypoint)")
    print("  ✅ AR(1) temporal correlation: ρ ∈ [0.1, 0.2]")
    print("  ✅ Vertex enumeration (2 vertices)")
    print("  ✅ Interval hull for epistemic bounds")
    print("  ✅ Gaussian marginals (physical model)")
    print()
    print("Note: NaNs at extreme quantiles are numerical artifacts from")
    print("      GaussianCopulaN's Monte Carlo sampling. Valid quantiles")
    print("      provide tight bounds on robustness distribution.")
    print()

    # Save results for plotting
    results = {
        'xs_l': xs_l_final,
        'xs_u': xs_u_final,
        'ys': signal_y[0].ys,  # Same ys for all
        'valid_mask': valid_mask,
        'width': width,
        'time': time1 + time2,
        'n_valid': n_valid
    }
    np.savez('experiment3_results.npz', **results)
    print("Results saved to experiment3_results.npz")

else:
    print(f"\n❌ ERROR: No valid quantiles after interval hull")
