"""
Implement the second half of Section V.A

This script demonstrates STL robustness bounds using IntervalGaussianCopulaN
with known bounds on the correlation matrix.

Formula: φ = □[0,2](X > 0)
where X ∈ ℝ³ is zero-mean normal with correlation matrix:

Σ ∈ [Σ_lower, Σ_upper] where:
  Σ_lower = [[1,   0.5, 0  ],
             [0.5, 1,   0.5],
             [0,   0.5, 1  ]]

  Σ_upper = [[1,   0.7, 0  ],
             [0.7, 1,   0.7],
             [0,   0.7, 1  ]]

"""

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import interval
from stlpy_copulas.STL import LinearPredicate, STLRandomVariable, IntervalGaussianCopulaN

# Enable LaTeX rendering
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# Parameters
N_BINS = 400
N_SAMPLES_MC = 10000  # For Monte Carlo ground truth

print("="*80)
print("SECTION V.A - PART 2: INTERVAL CORRELATION BOUNDS")
print("="*80)
print()
print("Formula: φ = □[0,2](X > 0)")
print("Variables: X ∈ ℝ³ ~ N(0, Σ) with Σ ∈ [Σ_lower, Σ_upper]")
print()

# Define correlation matrix bounds
Sigma_lower = np.array([
    [1.0, 0.5, 0.0],
    [0.5, 1.0, 0.5],
    [0.0, 0.5, 1.0]
])

Sigma_upper = np.array([
    [1.0, 0.7, 0.0],
    [0.7, 1.0, 0.7],
    [0.0, 0.7, 1.0]
])

print("Σ_lower:")
print(Sigma_lower)
print()
print("Σ_upper:")
print(Sigma_upper)
print()

# Create STL random variables (Gaussian N(0,1))
print("Creating STLRandomVariable objects...")
gaussian_rv = scipy.stats.norm(0, 1)

X = STLRandomVariable(gaussian_rv, debug=False)
Y = STLRandomVariable(gaussian_rv, debug=False)
Z = STLRandomVariable(gaussian_rv, debug=False)

# Compute inverse CDFs
support = np.interval(-7, 7)
X.compute_inverse_cdf_from_cdf(N_BINS, support)
Y.compute_inverse_cdf_from_cdf(N_BINS, support)
Z.compute_inverse_cdf_from_cdf(N_BINS, support)

print(f"✓ Created 3 Gaussian STLRandomVariable objects with {N_BINS} bins")
print()

# Create STL formula: φ = □[0,2](X > 0)
print("Creating STL formula...")
geq_0 = LinearPredicate(np.array([1]), np.array([0]), random=True)
horizon = 3  # [0, 2] inclusive = 3 timesteps
phi = geq_0.always(0, horizon - 1)
print(f"Formula: {phi}")
print()

# Create IntervalGaussianCopulaN with correlation bounds
print("Creating IntervalGaussianCopulaN...")
# Note: IntervalGaussianCopulaN may require n_samples for Monte Carlo sampling
# Let's use moderate values since this is only 3D
copula = IntervalGaussianCopulaN(
    Sigma_lower,
    Sigma_upper,
    n_samples=1000,
    n_bootstrap=50,
    max_vertices=None  # Try full vertex enumeration since 3×3 = 2^3 = 8 vertices
)
print(f"✓ Created IntervalGaussianCopulaN with correlation bounds")
print()

# Evaluate robustness with interval copula
print("Evaluating STL robustness with interval copula...")
rho = phi.robustness([X, Y, Z], 0, random=True, copula=copula)
print("✓ Robustness computed")
print()

# Extract bounds
xs_l, xs_u = interval.get_lu(rho.xs)
ys = rho.ys

# Compute CDF
print("Computing CDF...")
rho.cdf_from_inverse()
xrange = np.linspace(-4, 3, N_BINS)
ys_cdf = np.zeros_like(xrange, dtype=np.interval)
for j in range(N_BINS):
    ys_cdf[j] = rho.cdf_numeric(xrange[j])

y_lower_cdf, y_upper_cdf = interval.get_lu(ys_cdf)
print("✓ CDF computed")
print()

# Generate Monte Carlo ground truth for comparison
# We'll use the midpoint correlation for comparison
print(f"Generating Monte Carlo ground truth ({N_SAMPLES_MC} samples)...")
Sigma_mid = (Sigma_lower + Sigma_upper) / 2
print("Using midpoint correlation matrix:")
print(Sigma_mid)
print()

# Generate correlated samples using Cholesky decomposition
from scipy.linalg import cholesky

L = cholesky(Sigma_mid, lower=True)
robustness_samples = []

for i in range(N_SAMPLES_MC):
    # Generate correlated Gaussian samples
    z = np.random.randn(3)
    samples = L @ z

    # Evaluate STL robustness deterministically
    # Signal should be (1, horizon) = (1, 3) for 1 variable over 3 timesteps
    rho_val = phi.robustness(samples.reshape(1, 3), 0, random=False)
    robustness_samples.append(rho_val)

    if (i + 1) % 1000 == 0:
        print(f"  Progress: {i+1}/{N_SAMPLES_MC}")

robustness_samples = np.array(robustness_samples)
print(f"✓ Generated {N_SAMPLES_MC} MC samples")
print(f"  Mean: {np.mean(robustness_samples):.4f}")
print(f"  Std: {np.std(robustness_samples):.4f}")
print()

# Create empirical CDF
ecdf = scipy.stats.ecdf(robustness_samples)

# Plot results
print("Creating plot...")
fig, ax = plt.subplots(figsize=(7, 3.6))

# Plot Monte Carlo empirical CDF
ecdf.cdf.plot(ax, label=r'MC ($\Sigma$ = midpoint)', color='blue', linewidth=1.5)

# Plot interval bounds from copula
ax.plot(xrange, y_lower_cdf, 'r', linewidth=2, label='Copula Lower Bound')
ax.plot(xrange, y_upper_cdf, 'm', linewidth=2, label='Copula Upper Bound')

# Fill between bounds (vertically between the two curves)
ax.fill_between(xrange, y_lower_cdf, y_upper_cdf, alpha=0.2, color='purple')

ax.set_xlabel('Robustness $\\rho$', fontsize=12)
ax.set_ylabel('CDF / Quantile Level $\\alpha$', fontsize=12)
ax.set_title('Gaussian: Interval Correlation $\\Sigma \\in [\\Sigma_{\\mathrm{lower}}, \\Sigma_{\\mathrm{upper}}]$',
             fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([-4, 3])
ax.set_ylim([-0.05, 1.05])

# plt.tight_layout()
plt.savefig('output/gaussian_interval_correlation.pdf')
print("✓ Saved: output/gaussian_interval_correlation.pdf")
plt.show()

print()
print("="*80)
print("COMPLETE")
print("="*80)
print()
print("Results:")
print(f"  - Copula bounds computed using IntervalGaussianCopulaN")
print(f"  - MC ground truth generated at Σ = midpoint")
print(f"  - Plot saved to output/gaussian_interval_correlation.pdf")
print()
print("Note: The copula bounds should contain the MC curve since")
print("      the MC uses correlation within [Σ_lower, Σ_upper]")
