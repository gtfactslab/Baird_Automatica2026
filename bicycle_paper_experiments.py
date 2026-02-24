"""
Four Experiments for Bicycle Scenario (Paper Version)

This script generates four experiments demonstrating different copula methods
for probabilistic STL robustness computation on a kinematic bicycle model:

1. β Distribution: FH bounds + MC independent ground truth
2. Gaussian + FH Only: Baseline conservative bounds
3. Gaussian + 2-Copula: GaussianCopula2D with ρ ∈ [0.1, 0.2]
4. Gaussian + 25-Copula: IntervalGaussianCopulaN with AR(1) ρ ∈ [0.1, 0.2]

Output:
- bicycle_four_experiments.pdf: 2×2 grid of robustness CDF plots
- bicycle_experiments_timing.txt: Computation times for each experiment

Author: Claude Code, Luke Baird
Date: 2026-01-07
"""

import numpy as np
import scipy.stats
from scipy.linalg import cholesky
import matplotlib.pyplot as plt
import time
import copy

import interval
from stlpy_copulas.STL import (
    LinearPredicate,
    STLRandomVariable,
    GaussianCopula2D,
    GaussianCopulaN,
    IntervalGaussianCopulaN,
    FrechetHoeffdingCopula
)

# Import from Bicycle module
from Bicycle import (
    Bicycle,
    simulate_nominal_trajectory,
    create_bicycle_stl_formula,
    create_bicycle_stl_formula_2copula,
    construct_ar1_correlation,
    X0, U_NOMINAL, T_HORIZON, CORRIDOR_WIDTH, WAYPOINT_Y, WAYPOINT_TIME_WINDOW,
    DT, L_WHEELBASE, SIGMA_Y_BASE
)

# LaTeX rendering for publication-quality plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['font.size'] = 10

# ============================================================================
# PARAMETERS
# ============================================================================

NBINS = 50                 # CDF discretization bins
N_SAMPLES = 10000          # Monte Carlo samples
N_BOOTSTRAP = 100          # Bootstrap resamples
# SIGMA_Y_BASE imported from Bicycle module

# Random seed for reproducibility
np.random.seed(42)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def build_signal_beta(nominal_traj, T, alpha, beta, nbins=NBINS):
    """
    Build STL signal with interval-valued CDFs using β distribution.

    Similar to build_signal_with_uncertainty but uses β distribution instead
    of Gaussian. Uncertainty grows as √(t+1) for open-loop propagation.

    Args:
        nominal_traj (np.ndarray): (4 × T+1) nominal trajectory
        T (int): Time horizon [steps]
        alpha (float): β distribution parameter α (shape parameter a in scipy)
        beta (float): β distribution parameter β (shape parameter b in scipy)
        nbins (int): Number of bins for CDF discretization

    Returns:
        signal_y (list): List of T+1 STLRandomVariable objects for y-position
    """
    signal_y = []

    for t in range(T + 1):
        # Uncertainty grows as √(t + 1) for open-loop propagation
        sqrt_t_factor = np.sqrt(t + 1)

        # Y-position uncertainty (lateral deviation)
        std_y = SIGMA_Y_BASE * sqrt_t_factor
        mean_y = nominal_traj[1, t]

        # β distribution parameters
        # For β(a, b) on [0, 1], E[X] = a/(a+b), Var[X] = ab/((a+b)²(a+b+1))
        # Use loc and scale to shift/scale to desired mean and variance
        # Center around mean_y with spread proportional to std_y

        width = 6 * std_y  # Total width (6σ for ±3σ coverage)
        loc = mean_y - width * alpha / (alpha + beta)
        scale = width

        beta_rv = scipy.stats.beta(a=alpha, b=beta, loc=loc, scale=scale)
        rv_y = STLRandomVariable(beta_rv, debug=False)
        rv_y.compute_inverse_cdf_from_cdf(
            nbins,
            np.interval(mean_y - 3 * std_y, mean_y + 3 * std_y)
        )
        signal_y.append(rv_y)

    return signal_y


def evaluate_mc_independent_beta(phi, nominal_traj, T, alpha, beta, n_samples, n_bootstrap):
    """
    Evaluate probabilistic STL robustness using Monte Carlo with independent β-distributed disturbances.

    This represents "empirical truth" for the independent case (ρ=0).

    Args:
        phi (STLFormula): STL formula to evaluate
        nominal_traj (np.ndarray): (4 × T+1) nominal trajectory
        T (int): Time horizon [steps]
        alpha (float): β distribution parameter α
        beta (float): β distribution parameter β
        n_samples (int): Number of Monte Carlo samples
        n_bootstrap (int): Number of bootstrap resamples for CI

    Returns:
        rho_mc (object): Pseudo STLRandomVariable with empirical CDF
        samples (np.ndarray): Raw robustness samples for analysis
    """
    print(f"  Generating {n_samples} independent β-distributed trajectories...")
    robustness_samples = []

    for i in range(n_samples):
        # Sample T+1 independent β random variables
        W = np.zeros(T + 1)
        for t in range(T + 1):
            sqrt_t_factor = np.sqrt(t + 1)
            std_y = SIGMA_Y_BASE * sqrt_t_factor
            mean_y = nominal_traj[1, t]

            # β distribution for this timestep
            width = 6 * std_y
            loc = mean_y - width * alpha / (alpha + beta)
            scale = width

            # Sample from β distribution and subtract mean to get disturbance
            beta_sample = scipy.stats.beta.rvs(a=alpha, b=beta, loc=loc, scale=scale)
            W[t] = beta_sample - mean_y

        # Construct perturbed y-position trajectory
        y_perturbed = nominal_traj[1, :] + W
        y_signal = y_perturbed.reshape(1, -1)

        # Evaluate STL deterministically
        rho_i = phi.robustness(y_signal, 0, random=False)
        robustness_samples.append(rho_i)

        if (i + 1) % 1000 == 0:
            print(f"    Progress: {i+1}/{n_samples} samples")

    robustness_samples = np.array(robustness_samples)
    print(f"  ✓ Generated {n_samples} robustness samples")

    # Build empirical CDF with bootstrap confidence intervals
    print("  Computing empirical CDF with bootstrap confidence intervals...")
    quantiles = np.linspace(0, 1, 50)
    empirical_lower = []
    empirical_upper = []

    for q in quantiles:
        # Bootstrap percentile confidence interval
        bootstrap_vals = []
        for _ in range(n_bootstrap):
            resample = np.random.choice(robustness_samples,
                                       size=len(robustness_samples),
                                       replace=True)
            bootstrap_vals.append(np.percentile(resample, q*100))

        # 95% confidence interval
        empirical_lower.append(np.percentile(bootstrap_vals, 2.5))
        empirical_upper.append(np.percentile(bootstrap_vals, 97.5))

    empirical_lower = np.array(empirical_lower)
    empirical_upper = np.array(empirical_upper)

    print("  ✓ Built empirical CDF with 95% bootstrap confidence intervals")

    # Create pseudo STLRandomVariable for consistent interface
    rho_mc = type('obj', (object,), {
        'ys': quantiles,
        'xs': np.array([np.interval(l, u) for l, u in zip(empirical_lower, empirical_upper)])
    })()

    return rho_mc, robustness_samples


def evaluate_mc_independent_gaussian(phi, nominal_traj, T, n_samples, n_bootstrap):
    """
    Evaluate probabilistic STL robustness using Monte Carlo with independent Gaussian disturbances.

    This represents "empirical truth" for the independent case (ρ=0).

    Args:
        phi (STLFormula): STL formula to evaluate
        nominal_traj (np.ndarray): (4 × T+1) nominal trajectory
        T (int): Time horizon [steps]
        n_samples (int): Number of Monte Carlo samples
        n_bootstrap (int): Number of bootstrap resamples for CI

    Returns:
        rho_mc (object): Pseudo STLRandomVariable with empirical CDF
        samples (np.ndarray): Raw robustness samples for analysis
    """
    print(f"  Generating {n_samples} independent Gaussian trajectories...")
    robustness_samples = []

    for i in range(n_samples):
        # Sample T+1 independent Gaussian random variables
        W = np.zeros(T + 1)
        for t in range(T + 1):
            sqrt_t_factor = np.sqrt(t + 1)
            std_y = SIGMA_Y_BASE * sqrt_t_factor
            mean_y = nominal_traj[1, t]

            # Sample from Gaussian distribution
            gaussian_sample = np.random.normal(loc=mean_y, scale=std_y)
            W[t] = gaussian_sample - mean_y

        # Construct perturbed y-position trajectory
        y_perturbed = nominal_traj[1, :] + W
        y_signal = y_perturbed.reshape(1, -1)

        # Evaluate STL deterministically
        rho_i = phi.robustness(y_signal, 0, random=False)
        robustness_samples.append(rho_i)

        if (i + 1) % 1000 == 0:
            print(f"    Progress: {i+1}/{n_samples} samples")

    robustness_samples = np.array(robustness_samples)
    print(f"  ✓ Generated {n_samples} robustness samples")

    # Build empirical CDF with bootstrap confidence intervals
    print("  Computing empirical CDF with bootstrap confidence intervals...")
    quantiles = np.linspace(0, 1, 50)
    empirical_lower = []
    empirical_upper = []

    for q in quantiles:
        # Bootstrap percentile confidence interval
        bootstrap_vals = []
        for _ in range(n_bootstrap):
            resample = np.random.choice(robustness_samples,
                                       size=len(robustness_samples),
                                       replace=True)
            bootstrap_vals.append(np.percentile(resample, q*100))

        # 95% confidence interval
        empirical_lower.append(np.percentile(bootstrap_vals, 2.5))
        empirical_upper.append(np.percentile(bootstrap_vals, 97.5))

    empirical_lower = np.array(empirical_lower)
    empirical_upper = np.array(empirical_upper)

    print("  ✓ Built empirical CDF with 95% bootstrap confidence intervals")

    # Create pseudo STLRandomVariable for consistent interface
    rho_mc = type('obj', (object,), {
        'ys': quantiles,
        'xs': np.array([np.interval(l, u) for l, u in zip(empirical_lower, empirical_upper)])
    })()

    return rho_mc, robustness_samples


def evaluate_mc_correlated_gaussian(phi, nominal_traj, T, rho, n_samples, n_bootstrap):
    """
    Evaluate probabilistic STL robustness using Monte Carlo with AR(1) correlated Gaussian disturbances.

    This represents "empirical truth" for a fixed correlation ρ.

    Args:
        phi (STLFormula): STL formula to evaluate
        nominal_traj (np.ndarray): (4 × T+1) nominal trajectory
        T (int): Time horizon [steps]
        rho (float): AR(1) correlation coefficient
        n_samples (int): Number of Monte Carlo samples
        n_bootstrap (int): Number of bootstrap resamples for CI

    Returns:
        rho_mc (object): Pseudo STLRandomVariable with empirical CDF
        samples (np.ndarray): Raw robustness samples for analysis
    """

    print(f"  Generating {n_samples} AR(1) correlated Gaussian trajectories (ρ={rho})...")

    # Build AR(1) correlation matrix
    Sigma = construct_ar1_correlation(T + 1, rho=rho)
    L = cholesky(Sigma, lower=True)

    robustness_samples = []
    for i in range(n_samples):
        # Sample correlated Gaussian vector
        z = np.random.randn(T + 1)
        W = L @ z

        # Apply time-dependent scaling (√t growth)
        for t in range(T + 1):
            sqrt_t_factor = np.sqrt(t + 1)
            std_y = SIGMA_Y_BASE * sqrt_t_factor
            W[t] *= std_y

        # Construct perturbed y-position trajectory
        y_perturbed = nominal_traj[1, :] + W
        y_signal = y_perturbed.reshape(1, -1)

        # Evaluate STL deterministically
        rho_i = phi.robustness(y_signal, 0, random=False)
        robustness_samples.append(rho_i)

        if (i + 1) % 1000 == 0:
            print(f"    Progress: {i+1}/{n_samples} samples")

    robustness_samples = np.array(robustness_samples)
    print(f"  ✓ Generated {n_samples} robustness samples")

    # Build empirical CDF with bootstrap confidence intervals
    print("  Computing empirical CDF with bootstrap confidence intervals...")
    quantiles = np.linspace(0, 1, 50)
    empirical_lower = []
    empirical_upper = []

    for q in quantiles:
        # Bootstrap percentile confidence interval
        bootstrap_vals = []
        for _ in range(n_bootstrap):
            resample = np.random.choice(robustness_samples,
                                       size=len(robustness_samples),
                                       replace=True)
            bootstrap_vals.append(np.percentile(resample, q*100))

        # 95% confidence interval
        empirical_lower.append(np.percentile(bootstrap_vals, 2.5))
        empirical_upper.append(np.percentile(bootstrap_vals, 97.5))

    empirical_lower = np.array(empirical_lower)
    empirical_upper = np.array(empirical_upper)

    print("  ✓ Built empirical CDF with 95% bootstrap confidence intervals")

    # Create pseudo STLRandomVariable for consistent interface
    rho_mc = type('obj', (object,), {
        'ys': quantiles,
        'xs': np.array([np.interval(l, u) for l, u in zip(empirical_lower, empirical_upper)])
    })()

    return rho_mc, robustness_samples


# ============================================================================
# EXPERIMENT 1: β DISTRIBUTION
# ============================================================================

def experiment1_beta_distribution(traj, phi):
    """
    Experiment 1: β(α=5, β=2) distribution with FH bounds + MC independent.

    Returns:
        results (dict): {
            'fh': rho_fh,
            'mc': rho_mc,
            'time': elapsed_time
        }
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: β DISTRIBUTION (α=5, β=2)")
    print("=" * 80)

    start_time = time.time()

    # Build signal with β distribution
    print("Building signal with β distribution...")
    signal_y_beta = build_signal_beta(traj, T_HORIZON, alpha=5, beta=2, nbins=NBINS)
    print(f"✓ Created {len(signal_y_beta)} β-distributed STLRandomVariable objects")

    # Method 1: Fréchet-Hoeffding bounds
    print("\n[1.1] Evaluating with Fréchet-Hoeffding bounds...")
    copula_fh = FrechetHoeffdingCopula()
    rho_fh = phi.robustness(signal_y_beta, 0, random=True, copula=copula_fh)
    xs_l_fh, xs_u_fh = interval.get_lu(rho_fh.xs)
    valid_mask = np.isfinite(xs_l_fh) & np.isfinite(xs_u_fh)
    width_fh = np.mean((xs_u_fh - xs_l_fh)[valid_mask])
    print(f"  FH Width: {width_fh:.6f}")

    # Method 2: Monte Carlo with independent samples (empirical truth)
    print("\n[1.2] Evaluating with Monte Carlo (independent, ρ=0)...")
    rho_mc, samples_mc = evaluate_mc_independent_beta(
        phi, traj, T_HORIZON, alpha=5, beta=2,
        n_samples=N_SAMPLES, n_bootstrap=N_BOOTSTRAP
    )
    xs_l_mc, xs_u_mc = interval.get_lu(rho_mc.xs)
    valid_mask_mc = np.isfinite(xs_l_mc) & np.isfinite(xs_u_mc)
    width_mc = np.mean((xs_u_mc - xs_l_mc)[valid_mask_mc])
    print(f"  MC Width: {width_mc:.6f}")

    elapsed_time = time.time() - start_time
    print(f"\n✓ Experiment 1 completed in {elapsed_time:.2f} seconds")

    return {
        'fh': rho_fh,
        'mc': rho_mc,
        'time': elapsed_time
    }


# ============================================================================
# EXPERIMENT 2: GAUSSIAN + FH ONLY
# ============================================================================

def experiment2_gaussian_fh(traj, phi):
    """
    Experiment 2: Gaussian distribution with FH bounds + MC independent.

    Returns:
        results (dict): {
            'fh': rho_fh,
            'mc': rho_mc,
            'time': elapsed_time
        }
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: GAUSSIAN + FRÉCHET-HOEFFDING")
    print("=" * 80)

    start_time = time.time()

    # Build signal with Gaussian distribution
    print("Building signal with Gaussian distribution...")
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

    print(f"✓ Created {len(signal_y)} Gaussian STLRandomVariable objects")

    # Method 1: Fréchet-Hoeffding bounds
    print("\n[2.1] Evaluating with Fréchet-Hoeffding bounds...")
    copula_fh = FrechetHoeffdingCopula()
    rho_fh = phi.robustness(signal_y, 0, random=True, copula=copula_fh)
    xs_l_fh, xs_u_fh = interval.get_lu(rho_fh.xs)
    valid_mask = np.isfinite(xs_l_fh) & np.isfinite(xs_u_fh)
    width_fh = np.mean((xs_u_fh - xs_l_fh)[valid_mask])
    print(f"  FH Width: {width_fh:.6f}")

    # Method 2: Monte Carlo with independent samples (empirical truth)
    print("\n[2.2] Evaluating with Monte Carlo (independent, ρ=0)...")
    rho_mc, samples_mc = evaluate_mc_independent_gaussian(
        phi, traj, T_HORIZON,
        n_samples=N_SAMPLES, n_bootstrap=N_BOOTSTRAP
    )
    xs_l_mc, xs_u_mc = interval.get_lu(rho_mc.xs)
    valid_mask_mc = np.isfinite(xs_l_mc) & np.isfinite(xs_u_mc)
    width_mc = np.mean((xs_u_mc - xs_l_mc)[valid_mask_mc])
    print(f"  MC Width: {width_mc:.6f}")

    elapsed_time = time.time() - start_time
    print(f"\n✓ Experiment 2 completed in {elapsed_time:.2f} seconds")

    return {
        'fh': rho_fh,
        'mc': rho_mc,
        'time': elapsed_time
    }


# ============================================================================
# EXPERIMENT 3: GAUSSIAN + 2-COPULA
# ============================================================================

def experiment3_gaussian_2copula(traj, phi_original):
    """
    Experiment 3: Gaussian distribution with pairwise 2-copula (ρ ∈ [0.1, 0.2]).

    Uses GaussianCopula2D with a special formula structure that separates temporal
    operators from predicate conjunction. This allows proper modeling of temporal
    correlation without incorrectly applying the copula to predicates at the same timestep.

    To handle the interval ρ ∈ [0.1, 0.2], we evaluate at both endpoints and take
    worst-case bounds: min of lower bounds, max of upper bounds.

    Formula structure:
        Original: always[0,T]((y >= -hw) AND (y <= hw)) AND eventually[...]((y >= y0) AND (y <= y1))
        2-copula: (always[0,T](y >= -hw) AND always[0,T](y <= hw)) AND (eventually[...](y >= y0) AND eventually[...](y <= y1))

    Returns:
        results (dict): {
            'rho': rho,
            'mc': rho_mc,
            'mc_samples': samples,
            'time': elapsed_time
        }
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: GAUSSIAN + PAIRWISE 2-COPULA (ρ ∈ [0.1, 0.2])")
    print("=" * 80)
    print("Method: GaussianCopula2D with Williamson-Downs algorithm")

    start_time = time.time()

    # Build signal with Gaussian distribution
    print("Building signal with Gaussian distribution...")
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

    print(f"✓ Created {len(signal_y)} Gaussian STLRandomVariable objects")

    # Create 2-copula compatible formula structure
    print("\n[3.1] Creating 2-copula compatible formula structure...")
    print("  Original: always[0,T]((y >= -hw) AND (y <= hw)) AND ...")
    print("  2-copula: (always(y >= -hw) AND always(y <= hw)) AND ...")
    phi_2copula = create_bicycle_stl_formula_2copula(
        T_HORIZON,
        CORRIDOR_WIDTH / 2,
        WAYPOINT_Y,
        WAYPOINT_TIME_WINDOW
    )
    print("  ✓ Created formula with separated temporal/conjunction structure")

    # Evaluate with GaussianCopula2D at both endpoints of ρ ∈ [0.1, 0.2]
    print("\n[3.2] Evaluating with GaussianCopula2D (ρ ∈ [0.1, 0.2])...")

    # Evaluate at ρ = 0.1
    print("  Evaluating at ρ = 0.1...")
    copula_01 = GaussianCopula2D(0.1)
    rho_01 = phi_2copula.robustness(signal_y, 0, random=True, copula=copula_01)
    xs_l_01, xs_u_01 = interval.get_lu(rho_01.xs)

    # Evaluate at ρ = 0.2
    print("  Evaluating at ρ = 0.2...")
    copula_02 = GaussianCopula2D(0.2)
    rho_02 = phi_2copula.robustness(signal_y, 0, random=True, copula=copula_02)
    xs_l_02, xs_u_02 = interval.get_lu(rho_02.xs)

    # Take worst-case bounds
    print("  Taking worst-case bounds...")
    xs_l = np.minimum(xs_l_01, xs_l_02)
    xs_u = np.maximum(xs_u_01, xs_u_02)

    # Create combined result by copying rho_01 and updating xs
    rho_2d = copy.deepcopy(rho_01)
    rho_2d.xs = np.array([np.interval(l, u) for l, u in zip(xs_l, xs_u)])

    valid_mask = np.isfinite(xs_l) & np.isfinite(xs_u)
    n_valid = np.sum(valid_mask)
    if n_valid > 0:
        width_01 = np.mean((xs_u_01 - xs_l_01)[valid_mask])
        width_02 = np.mean((xs_u_02 - xs_l_02)[valid_mask])
        width_2d = np.mean((xs_u - xs_l)[valid_mask])
        print(f"  Valid quantiles: {n_valid}/50")
        print(f"  Width at ρ=0.1: {width_01:.6f}")
        print(f"  Width at ρ=0.2: {width_02:.6f}")
        print(f"  Combined Width: {width_2d:.6f}")
    else:
        width_2d = np.nan
        print(f"  Valid quantiles: {n_valid}/50 (all invalid)")

    # Monte Carlo with correlated samples (ρ = 0.15)
    print("\n[3.3] Evaluating with Monte Carlo (AR(1) correlated, ρ = 0.15)...")
    rho_mc, samples_mc = evaluate_mc_correlated_gaussian(
        phi_2copula, traj, T_HORIZON, rho=0.15,
        n_samples=N_SAMPLES, n_bootstrap=N_BOOTSTRAP
    )
    xs_l_mc, xs_u_mc = interval.get_lu(rho_mc.xs)
    valid_mask_mc = np.isfinite(xs_l_mc) & np.isfinite(xs_u_mc)
    width_mc = np.mean((xs_u_mc - xs_l_mc)[valid_mask_mc])
    print(f"  MC Width: {width_mc:.6f}")

    elapsed_time = time.time() - start_time
    print(f"\n✓ Experiment 3 completed in {elapsed_time:.2f} seconds")

    return {
        'rho': rho_2d,
        'mc': rho_mc,
        'mc_samples': samples_mc,
        'time': elapsed_time
    }


# ============================================================================
# EXPERIMENT 4: GAUSSIAN + 25-COPULA
# ============================================================================

def experiment4_gaussian_25copula(traj, phi):
    """
    Experiment 4: Gaussian distribution with IntervalGaussianCopulaN (ρ ∈ [0.1, 0.2]).

    Uses full 25-dimensional copula with AR(1) correlation structure.

    Returns:
        results (dict): {
            'rho': rho,
            'time': elapsed_time
        }
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: GAUSSIAN + 25-COPULA (ρ ∈ [0.1, 0.2])")
    print("=" * 80)

    start_time = time.time()

    # Build signal with Gaussian distribution
    print("Building signal with Gaussian distribution...")
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

    print(f"✓ Created {len(signal_y)} Gaussian STLRandomVariable objects")

    # Create AR(1) correlation matrices
    print("\n[4.1] Creating AR(1) correlation matrices...")
    Sigma_lower = construct_ar1_correlation(T_HORIZON + 1, rho=0.1)
    Sigma_upper = construct_ar1_correlation(T_HORIZON + 1, rho=0.2)
    print(f"  ✓ Constructed {T_HORIZON + 1}×{T_HORIZON + 1} correlation matrices")

    # Evaluate with IntervalGaussianCopulaN
    print("\n[4.2] Evaluating with IntervalGaussianCopulaN...")
    copula = IntervalGaussianCopulaN(
        Sigma_lower, Sigma_upper,
        n_samples=N_SAMPLES,
        n_bootstrap=N_BOOTSTRAP
    )
    rho = phi.robustness(signal_y, 0, random=True, copula=copula)
    xs_l, xs_u = interval.get_lu(rho.xs)
    valid_mask = np.isfinite(xs_l) & np.isfinite(xs_u)
    width = np.mean((xs_u - xs_l)[valid_mask])
    print(f"  25-Copula Width: {width:.6f}")

    elapsed_time = time.time() - start_time
    print(f"\n✓ Experiment 4 completed in {elapsed_time:.2f} seconds")

    return {
        'rho': rho,
        'time': elapsed_time
    }


# ============================================================================
# PLOTTING
# ============================================================================

def plot_single_experiment(results, exp_num, title, colors, filename, xlim=None):
    """
    Create a single plot for one experiment.

    Args:
        results (dict): Experiment results
        exp_num (int): Experiment number (1, 2, or 3)
        title (str): Plot title
        colors (dict): Color scheme for the plot
        filename (str): Output filename
        xlim (tuple): Optional (xmin, xmax) for x-axis limits
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 2))

    if exp_num == 1:
        # Plot 1: β Distribution with FH and MC
        rho_fh = results['fh']
        xs_l_fh, xs_u_fh = interval.get_lu(rho_fh.xs)
        quantiles_fh = rho_fh.ys

        # Handle infinities - extend to xlim edges if provided (matching other experiments)
        xs_l_fh_plot = xs_l_fh.copy()
        xs_u_fh_plot = xs_u_fh.copy()

        if xlim is not None:
            # Use xlim boundaries for infinities
            xs_l_fh_plot[np.isinf(xs_l_fh_plot) & (xs_l_fh_plot < 0)] = xlim[0]
            xs_u_fh_plot[np.isinf(xs_u_fh_plot) & (xs_u_fh_plot > 0)] = xlim[1]
        else:
            # Fallback to +/- 0.5 from finite range
            finite_lower = xs_l_fh[np.isfinite(xs_l_fh)]
            finite_upper = xs_u_fh[np.isfinite(xs_u_fh)]
            if len(finite_lower) > 0:
                min_finite = np.min(finite_lower)
                xs_l_fh_plot[np.isinf(xs_l_fh_plot) & (xs_l_fh_plot < 0)] = min_finite - 0.5
            if len(finite_upper) > 0:
                max_finite = np.max(finite_upper)
                xs_u_fh_plot[np.isinf(xs_u_fh_plot) & (xs_u_fh_plot > 0)] = max_finite + 0.5

        ax.plot(xs_l_fh_plot, quantiles_fh, colors['fh'], linewidth=2, label='FH Lower')
        ax.plot(xs_u_fh_plot, quantiles_fh, colors['fh'], linewidth=2, linestyle='--', label='FH Upper')
        ax.fill_betweenx(quantiles_fh, xs_l_fh_plot, xs_u_fh_plot,
                         alpha=0.3, color=colors['fh'])

        rho_mc = results['mc']
        xs_l_mc, xs_u_mc = interval.get_lu(rho_mc.xs)
        quantiles_mc = rho_mc.ys

        ax.plot(xs_l_mc, quantiles_mc, colors['mc'], linewidth=2, label='MC Lower')
        ax.plot(xs_u_mc, quantiles_mc, colors['mc'], linewidth=2, linestyle='--', label='MC Upper')
        ax.fill_betweenx(quantiles_mc, xs_l_mc, xs_u_mc,
                         alpha=0.3, color=colors['mc'])

    elif exp_num == 2:
        # Plot 2: Gaussian + FH and MC
        rho_fh = results['fh']
        xs_l_fh, xs_u_fh = interval.get_lu(rho_fh.xs)
        quantiles_fh = rho_fh.ys

        # Handle infinities - extend to xlim edges if provided
        xs_l_fh_plot = xs_l_fh.copy()
        xs_u_fh_plot = xs_u_fh.copy()

        if xlim is not None:
            # Use xlim boundaries for infinities
            xs_l_fh_plot[np.isinf(xs_l_fh_plot) & (xs_l_fh_plot < 0)] = xlim[0]
            xs_u_fh_plot[np.isinf(xs_u_fh_plot) & (xs_u_fh_plot > 0)] = xlim[1]
        else:
            # Fallback to +/- 0.5 from finite range
            finite_lower = xs_l_fh[np.isfinite(xs_l_fh)]
            finite_upper = xs_u_fh[np.isfinite(xs_u_fh)]
            if len(finite_lower) > 0:
                min_finite = np.min(finite_lower)
                xs_l_fh_plot[np.isinf(xs_l_fh_plot) & (xs_l_fh_plot < 0)] = min_finite - 0.5
            if len(finite_upper) > 0:
                max_finite = np.max(finite_upper)
                xs_u_fh_plot[np.isinf(xs_u_fh_plot) & (xs_u_fh_plot > 0)] = max_finite + 0.5

        ax.plot(xs_l_fh_plot, quantiles_fh, colors['fh'], linewidth=2, label='FH Lower')
        ax.plot(xs_u_fh_plot, quantiles_fh, colors['fh'], linewidth=2, linestyle='--', label='FH Upper')
        ax.fill_betweenx(quantiles_fh, xs_l_fh_plot, xs_u_fh_plot,
                         alpha=0.3, color=colors['fh'])

        rho_mc = results['mc']
        xs_l_mc, xs_u_mc = interval.get_lu(rho_mc.xs)
        quantiles_mc = rho_mc.ys

        ax.plot(xs_l_mc, quantiles_mc, colors['mc'], linewidth=2, label='MC Lower')
        ax.plot(xs_u_mc, quantiles_mc, colors['mc'], linewidth=2, linestyle='--', label='MC Upper')
        ax.fill_betweenx(quantiles_mc, xs_l_mc, xs_u_mc,
                         alpha=0.3, color=colors['mc'])

    elif exp_num == 3:
        # Plot 3: Gaussian + AR(1) Correlation with Interval and MC
        # Option D: Plot interval envelope + MC median + bootstrap CI shading

        # Plot interval envelope (epistemic uncertainty from ρ ∈ [0.1, 0.2])
        rho_env = results['rho']
        xs_l_env, xs_u_env = interval.get_lu(rho_env.xs)
        quantiles_env = rho_env.ys

        ax.plot(xs_l_env, quantiles_env, colors['interval'], linewidth=2,
                label='Interval Lower (epistemic)')
        ax.plot(xs_u_env, quantiles_env, colors['interval'], linewidth=2, linestyle='--',
                label='Interval Upper (epistemic)')
        ax.fill_betweenx(quantiles_env, xs_l_env, xs_u_env,
                         alpha=0.3, color=colors['interval'])

        # Compute MC median (point estimate at ρ=0.15)
        mc_samples = results['mc_samples']
        quantiles_mc = rho_env.ys  # Use same quantile levels
        mc_median = np.percentile(mc_samples, quantiles_mc * 100)

        # Plot MC mean (empirical CDF) as solid line
        ax.plot(mc_median, quantiles_mc, colors['mc'], linewidth=1.0,
                label=r'MC Mean ($\rho=0.15$)', zorder=10)

    ax.axvline(x=0, color='red', linestyle='-', linewidth=1.5, alpha=0.7)
    ax.set_xlabel(r'Robustness $\rho$', fontsize=11)
    ax.set_ylabel(r'CDF / Quantile Level $\alpha$', fontsize=11)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    if xlim is not None:
        ax.set_xlim(xlim)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', backend='pdf')
    plt.close()


def plot_three_experiments(results1, results2, results3):
    """
    Create 1×3 subplot figure with three experiments.
    Also saves individual plots for LaTeX inclusion.

    Args:
        results1 (dict): Experiment 1 results
        results2 (dict): Experiment 2 results
        results3 (dict): Experiment 3 results
    """
    print("\n" + "=" * 80)
    print("GENERATING FIGURES")
    print("=" * 80)

    # Extract all data first to compute global x-axis range
    # Experiment 1
    rho_fh1 = results1['fh']
    xs_l_fh1, xs_u_fh1 = interval.get_lu(rho_fh1.xs)
    valid_fh1 = np.isfinite(xs_l_fh1) & np.isfinite(xs_u_fh1)

    rho_mc1 = results1['mc']
    xs_l_mc1, xs_u_mc1 = interval.get_lu(rho_mc1.xs)
    valid_mc1 = np.isfinite(xs_l_mc1) & np.isfinite(xs_u_mc1)

    # Experiment 2
    rho_fh2 = results2['fh']
    xs_l_fh2, xs_u_fh2 = interval.get_lu(rho_fh2.xs)
    valid_fh2 = np.isfinite(xs_l_fh2) & np.isfinite(xs_u_fh2)

    # Experiment 3
    rho_env3 = results3['rho']
    xs_l_env3, xs_u_env3 = interval.get_lu(rho_env3.xs)
    valid_env3 = np.isfinite(xs_l_env3) & np.isfinite(xs_u_env3)

    # Extract MC data for Experiment 3
    mc_samples3 = results3['mc_samples']
    rho_mc3 = results3['mc']
    xs_l_mc3, xs_u_mc3 = interval.get_lu(rho_mc3.xs)
    mc_median3 = np.percentile(mc_samples3, rho_env3.ys * 100)

    # Compute global x-axis range for alignment (especially for plots 2 and 3)
    # Use finite values only for computing range
    finite_xs2 = np.concatenate([
        xs_l_fh2[valid_fh2], xs_u_fh2[valid_fh2]
    ])
    finite_xs3 = np.concatenate([
        xs_l_env3[valid_env3], xs_u_env3[valid_env3]
    ])

    x_min_finite = min(np.min(finite_xs2), np.min(finite_xs3))
    x_max_finite = max(np.max(finite_xs2), np.max(finite_xs3))

    # Add margin and extend below minimum for -inf values
    x_margin = 0.1 * (x_max_finite - x_min_finite)
    x_min_global = x_min_finite - x_margin - 0.5  # Extra space for -inf
    x_max_global = x_max_finite + x_margin

    # Create combined 1×3 figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 2))
    fig.suptitle(r'Kinematic Bicycle Model: Probabilistic STL Robustness Analysis',
                 fontsize=14, fontweight='bold', y=1.02)

    # ========================================================================
    # Plot 1: β Distribution (left)
    # ========================================================================
    ax = axes[0]

    # Handle infinities in FH bounds
    xs_l_fh1_plot = xs_l_fh1.copy()
    xs_u_fh1_plot = xs_u_fh1.copy()
    finite_lower1 = xs_l_fh1[np.isfinite(xs_l_fh1)]
    finite_upper1 = xs_u_fh1[np.isfinite(xs_u_fh1)]
    if len(finite_lower1) > 0:
        min_finite1 = np.min(finite_lower1)
        xs_l_fh1_plot[np.isinf(xs_l_fh1_plot) & (xs_l_fh1_plot < 0)] = min_finite1 - 0.5
    if len(finite_upper1) > 0:
        max_finite1 = np.max(finite_upper1)
        xs_u_fh1_plot[np.isinf(xs_u_fh1_plot) & (xs_u_fh1_plot > 0)] = max_finite1 + 0.5

    ax.plot(xs_l_fh1_plot, rho_fh1.ys, 'orange', linewidth=2, label='FH Lower')
    ax.plot(xs_u_fh1_plot, rho_fh1.ys, 'orange', linewidth=2, linestyle='--', label='FH Upper')
    ax.fill_betweenx(rho_fh1.ys, xs_l_fh1_plot, xs_u_fh1_plot,
                     alpha=0.3, color='orange')

    ax.plot(xs_l_mc1, rho_mc1.ys, 'blue', linewidth=2, label='MC Lower')
    ax.plot(xs_u_mc1, rho_mc1.ys, 'blue', linewidth=2, linestyle='--', label='MC Upper')
    ax.fill_betweenx(rho_mc1.ys, xs_l_mc1, xs_u_mc1,
                     alpha=0.2, color='blue')

    ax.axvline(x=0, color='red', linestyle='-', linewidth=1.5, alpha=0.7)
    ax.set_xlabel(r'Robustness $\rho$', fontsize=11)
    ax.set_ylabel(r'CDF / Quantile Level $\alpha$', fontsize=11)
    ax.set_title(r'$\beta$ Distribution ($\alpha=5, \beta=2$)', fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # ========================================================================
    # Plot 2: Gaussian + FH (center)
    # ========================================================================
    ax = axes[1]

    # Handle infinities in FH bounds
    xs_l_fh2_plot = xs_l_fh2.copy()
    xs_u_fh2_plot = xs_u_fh2.copy()
    finite_lower2 = xs_l_fh2[np.isfinite(xs_l_fh2)]
    finite_upper2 = xs_u_fh2[np.isfinite(xs_u_fh2)]
    if len(finite_lower2) > 0:
        min_finite2 = np.min(finite_lower2)
        xs_l_fh2_plot[np.isinf(xs_l_fh2_plot) & (xs_l_fh2_plot < 0)] = min_finite2 - 0.5
    if len(finite_upper2) > 0:
        max_finite2 = np.max(finite_upper2)
        xs_u_fh2_plot[np.isinf(xs_u_fh2_plot) & (xs_u_fh2_plot > 0)] = max_finite2 + 0.5

    ax.plot(xs_l_fh2_plot, rho_fh2.ys, 'green', linewidth=2, label='FH Lower')
    ax.plot(xs_u_fh2_plot, rho_fh2.ys, 'green', linewidth=2, linestyle='--', label='FH Upper')
    ax.fill_betweenx(rho_fh2.ys, xs_l_fh2_plot, xs_u_fh2_plot,
                     alpha=0.3, color='green')

    ax.axvline(x=0, color='red', linestyle='-', linewidth=1.5, alpha=0.7)
    ax.set_xlabel(r'Robustness $\rho$', fontsize=11)
    ax.set_ylabel(r'CDF / Quantile Level $\alpha$', fontsize=11)
    ax.set_title(r'Gaussian: Fr\'echet-Hoeffding Only', fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    ax.set_xlim([x_min_global - x_margin, x_max_global + x_margin])

    # ========================================================================
    # Plot 3: Gaussian + Pairwise Correlation (right)
    # ========================================================================
    ax = axes[2]

    # Plot interval envelope (epistemic from ρ)
    ax.plot(xs_l_env3, rho_env3.ys, 'purple', linewidth=2, label='Interval Lower')
    ax.plot(xs_u_env3, rho_env3.ys, 'purple', linewidth=2, linestyle='--', label='Interval Upper')
    ax.fill_betweenx(rho_env3.ys, xs_l_env3, xs_u_env3,
                     alpha=0.3, color='purple')

    # Plot MC median (point estimate at ρ=0.15)
    ax.plot(mc_median3, rho_env3.ys, 'blue', linewidth=1.0,
            label=r'MC ($\rho=0.15$)', zorder=10)

    # Plot MC bootstrap CI as light shading
    ax.fill_betweenx(rho_env3.ys, xs_l_mc3, xs_u_mc3,
                     alpha=0.15, color='blue')

    ax.axvline(x=0, color='red', linestyle='-', linewidth=1.5, alpha=0.7)
    ax.set_xlabel(r'Robustness $\rho$', fontsize=11)
    ax.set_ylabel(r'CDF / Quantile Level $\alpha$', fontsize=11)
    ax.set_title(r'Gaussian: Pairwise Corr. ($\varrho \in [0.1, 0.2]$)', fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    ax.set_xlim([x_min_global - x_margin, x_max_global + x_margin])

    # Save combined figure
    plt.tight_layout()
    plt.savefig('output/bicycle_three_experiments.pdf', dpi=300, bbox_inches='tight', backend='pdf')
    print("✓ Saved: output/bicycle_three_experiments.pdf")
    plt.close()

    # Save individual plots
    print("\nGenerating individual plots...")

    # Use same x-axis limits for all experiments to align ρ=0 line
    aligned_xlim = [-2.5, 3.5]

    plot_single_experiment(results1, 1,
                          r'$\beta$ Distribution ($\alpha=5, \beta=2$)',
                          {'fh': 'orange', 'mc': 'blue'},
                          'output/bicycle_exp1_beta.pdf',
                          xlim=aligned_xlim)
    print("✓ Saved: output/bicycle_exp1_beta.pdf")

    plot_single_experiment(results2, 2,
                          r'Gaussian Distribution',
                          {'fh': 'magenta', 'mc': 'blue'},
                          'output/bicycle_exp2_gaussian_fh.pdf',
                          xlim=aligned_xlim)
    print("✓ Saved: output/bicycle_exp2_gaussian_fh.pdf")

    plot_single_experiment(results3, 3,
                          r'Gaussian: Corr. ($\varrho \in [0.1, 0.2]$)',
                          {'interval': 'green', 'mc': 'blue'},
                          'output/bicycle_exp3_gaussian_pairwise.pdf',
                          xlim=aligned_xlim)
    print("✓ Saved: output/bicycle_exp3_gaussian_pairwise.pdf")


def plot_four_experiments(results1, results2, results3, results4):
    """
    Create 2×2 subplot figure with all four experiments.

    Args:
        results1 (dict): Experiment 1 results
        results2 (dict): Experiment 2 results
        results3 (dict): Experiment 3 results
        results4 (dict): Experiment 4 results
    """
    print("\n" + "=" * 80)
    print("GENERATING 2×2 FIGURE")
    print("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(r'Kinematic Bicycle Model: Probabilistic STL Robustness Analysis',
                 fontsize=14, fontweight='bold', y=0.995)

    # Common y-axis limits (quantile levels)
    quantile_range = [0, 1]

    # ========================================================================
    # Plot 1: β Distribution (top-left)
    # ========================================================================
    ax = axes[0, 0]

    # FH bounds
    rho_fh = results1['fh']
    xs_l_fh, xs_u_fh = interval.get_lu(rho_fh.xs)
    valid_fh = np.isfinite(xs_l_fh) & np.isfinite(xs_u_fh)
    quantiles_fh = rho_fh.ys

    ax.plot(xs_l_fh[valid_fh], quantiles_fh[valid_fh], 'orange', linewidth=2, label='FH Lower')
    ax.plot(xs_u_fh[valid_fh], quantiles_fh[valid_fh], 'orange', linewidth=2, linestyle='--', label='FH Upper')
    ax.fill_betweenx(quantiles_fh[valid_fh], xs_l_fh[valid_fh], xs_u_fh[valid_fh],
                     alpha=0.3, color='orange')

    # MC independent
    rho_mc = results1['mc']
    xs_l_mc, xs_u_mc = interval.get_lu(rho_mc.xs)
    valid_mc = np.isfinite(xs_l_mc) & np.isfinite(xs_u_mc)
    quantiles_mc = rho_mc.ys

    ax.plot(xs_l_mc[valid_mc], quantiles_mc[valid_mc], 'blue', linewidth=2, label='MC Lower')
    ax.plot(xs_u_mc[valid_mc], quantiles_mc[valid_mc], 'blue', linewidth=2, linestyle='--', label='MC Upper')
    ax.fill_betweenx(quantiles_mc[valid_mc], xs_l_mc[valid_mc], xs_u_mc[valid_mc],
                     alpha=0.2, color='blue')

    ax.axvline(x=0, color='red', linestyle='-', linewidth=1.5, alpha=0.7)
    ax.set_xlabel(r'Robustness $\rho$', fontsize=11)
    ax.set_ylabel(r'CDF / Quantile Level $\alpha$', fontsize=11)
    ax.set_title(r'(a) $\beta$ Distribution ($\alpha=5, \beta=2$)', fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(quantile_range)

    # ========================================================================
    # Plot 2: Gaussian + FH (top-right)
    # ========================================================================
    ax = axes[0, 1]

    rho_fh = results2['fh']
    xs_l_fh, xs_u_fh = interval.get_lu(rho_fh.xs)
    valid_fh = np.isfinite(xs_l_fh) & np.isfinite(xs_u_fh)
    quantiles_fh = rho_fh.ys

    ax.plot(xs_l_fh[valid_fh], quantiles_fh[valid_fh], 'green', linewidth=2, label='FH Lower')
    ax.plot(xs_u_fh[valid_fh], quantiles_fh[valid_fh], 'green', linewidth=2, linestyle='--', label='FH Upper')
    ax.fill_betweenx(quantiles_fh[valid_fh], xs_l_fh[valid_fh], xs_u_fh[valid_fh],
                     alpha=0.3, color='green')

    ax.axvline(x=0, color='red', linestyle='-', linewidth=1.5, alpha=0.7)
    ax.set_xlabel(r'Robustness $\rho$', fontsize=11)
    ax.set_ylabel(r'CDF / Quantile Level $\alpha$', fontsize=11)
    ax.set_title(r'(b) Gaussian: Fr\'echet-Hoeffding Only', fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(quantile_range)

    # ========================================================================
    # Plot 3: Gaussian + Pairwise Correlation (bottom-left)
    # ========================================================================
    ax = axes[1, 0]

    rho_env = results3['rho']
    xs_l_env, xs_u_env = interval.get_lu(rho_env.xs)
    valid_env = np.isfinite(xs_l_env) & np.isfinite(xs_u_env)
    quantiles_env = rho_env.ys

    ax.plot(xs_l_env[valid_env], quantiles_env[valid_env], 'purple', linewidth=2, label='Lower')
    ax.plot(xs_u_env[valid_env], quantiles_env[valid_env], 'purple', linewidth=2, linestyle='--', label='Upper')
    ax.fill_betweenx(quantiles_env[valid_env], xs_l_env[valid_env], xs_u_env[valid_env],
                     alpha=0.3, color='purple')

    ax.axvline(x=0, color='red', linestyle='-', linewidth=1.5, alpha=0.7)
    ax.set_xlabel(r'Robustness $\rho$', fontsize=11)
    ax.set_ylabel(r'CDF / Quantile Level $\alpha$', fontsize=11)
    ax.set_title(r'(c) Gaussian: Pairwise Corr. ($\varrho \in [0.1, 0.2]$)', fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(quantile_range)

    # ========================================================================
    # Plot 4: Gaussian + 25-Copula (bottom-right)
    # ========================================================================
    ax = axes[1, 1]

    rho = results4['rho']
    xs_l, xs_u = interval.get_lu(rho.xs)
    valid = np.isfinite(xs_l) & np.isfinite(xs_u)
    quantiles = rho.ys

    ax.plot(xs_l[valid], quantiles[valid], 'teal', linewidth=2, label='Lower')
    ax.plot(xs_u[valid], quantiles[valid], 'teal', linewidth=2, linestyle='--', label='Upper')
    ax.fill_betweenx(quantiles[valid], xs_l[valid], xs_u[valid],
                     alpha=0.3, color='teal')

    ax.axvline(x=0, color='red', linestyle='-', linewidth=1.5, alpha=0.7)
    ax.set_xlabel(r'Robustness $\rho$', fontsize=11)
    ax.set_ylabel(r'CDF / Quantile Level $\alpha$', fontsize=11)
    ax.set_title(r'(d) Gaussian: 25-Copula ($\varrho \in [0.1, 0.2]$)', fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(quantile_range)

    # Save figure
    plt.tight_layout()
    plt.savefig('output/bicycle_four_experiments.pdf', dpi=300, bbox_inches='tight', backend='pdf')
    print("✓ Saved: output/bicycle_four_experiments.pdf")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Main execution: Run all four experiments and generate outputs.
    """
    print("\n" + "=" * 80)
    print("BICYCLE MODEL: FOUR EXPERIMENTS FOR PAPER")
    print("=" * 80)
    print(f"\nParameters:")
    print(f"  Time horizon: T = {T_HORIZON} steps ({T_HORIZON * DT:.1f} seconds)")
    print(f"  Discretization: dt = {DT} s")
    print(f"  CDF bins: {NBINS}")
    print(f"  Monte Carlo samples: {N_SAMPLES}")
    print(f"  Bootstrap resamples: {N_BOOTSTRAP}")

    # Generate nominal trajectory (shared by all experiments)
    print("\n" + "=" * 80)
    print("GENERATING NOMINAL TRAJECTORY")
    print("=" * 80)
    traj = simulate_nominal_trajectory(X0, U_NOMINAL, T_HORIZON)
    print(f"✓ Generated trajectory with {T_HORIZON + 1} timesteps")
    print(f"  Final position: ({traj[0, -1]:.2f}, {traj[1, -1]:.2f}) m")

    # Create STL formula (shared by all experiments)
    print("\n" + "=" * 80)
    print("CREATING STL FORMULA")
    print("=" * 80)
    phi, _, _ = create_bicycle_stl_formula(
        T_HORIZON,
        CORRIDOR_WIDTH / 2,
        WAYPOINT_Y,
        WAYPOINT_TIME_WINDOW
    )
    print(f"✓ Formula: φ = G[0,{T_HORIZON}](corridor) ∧ F[{T_HORIZON - WAYPOINT_TIME_WINDOW},{T_HORIZON}](waypoint)")

    # Run first three experiments (skip 25-copula for now - too slow)
    total_start_time = time.time()

    results1 = experiment1_beta_distribution(traj, phi)
    results2 = experiment2_gaussian_fh(traj, phi)
    results3 = experiment3_gaussian_2copula(traj, phi)
    # results4 = experiment4_gaussian_25copula(traj, phi)  # SKIPPED - handle separately

    total_time = time.time() - total_start_time

    # Generate plots (3 experiments only)
    plot_three_experiments(results1, results2, results3)

    # Write timing results
    print("\n" + "=" * 80)
    print("WRITING TIMING RESULTS")
    print("=" * 80)

    with open('bicycle_experiments_timing.txt', 'w') as f:
        f.write("Bicycle Model: Three Experiments Timing Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Experiment 1 (β Distribution):     {results1['time']:8.2f} seconds\n")
        f.write(f"Experiment 2 (Gaussian + FH):      {results2['time']:8.2f} seconds\n")
        f.write(f"Experiment 3 (Gaussian + Pairwise):{results3['time']:8.2f} seconds\n")
        f.write("-" * 60 + "\n")
        f.write(f"Total:                             {total_time:8.2f} seconds\n")
        f.write("\nNote: Experiment 4 (25-Copula) skipped - handle separately\n")

    print("✓ Saved: bicycle_experiments_timing.txt")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nTiming:")
    print(f"  Experiment 1: {results1['time']:.2f} s")
    print(f"  Experiment 2: {results2['time']:.2f} s")
    print(f"  Experiment 3: {results3['time']:.2f} s")
    print(f"  Total: {total_time:.2f} s")

    print(f"\nGenerated files:")
    print("  - output/bicycle_three_experiments.pdf (combined 1×3 figure)")
    print("  - output/bicycle_exp1_beta.pdf (individual plot 1)")
    print("  - output/bicycle_exp2_gaussian_fh.pdf (individual plot 2)")
    print("  - output/bicycle_exp3_gaussian_pairwise.pdf (individual plot 3)")
    print("  - bicycle_experiments_timing.txt")
    print("\nNote: Experiment 4 (25-Copula) skipped - computationally intensive")

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
