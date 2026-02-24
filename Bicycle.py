"""
Bicycle Model for Probabilistic STL Verification

This module provides a Bicycle class encapsulating:
- Kinematic bicycle model dynamics
- Nominal trajectory simulation
- STL formula construction (corridor + waypoint)
- Signal building with uncertainty propagation
- AR(1) temporal correlation matrix construction

Literature Citations:
- Bicycle Model: Rajamani (2012) "Vehicle Dynamics and Control", Section 2.2
- AR(1) Temporal Correlation: Box & Jenkins (1976) "Time Series Analysis", Section 3.2
- STL Robustness: Donze & Maler (2010) "Robust Satisfaction of Temporal Logic"

Author: Claude Code, Luke Baird
Date: 2026-01-16
"""

import numpy as np
import scipy.stats

import interval
from stlpy_copulas.STL import LinearPredicate, STLRandomVariable


class Bicycle:
    """
    Kinematic bicycle model with STL verification support.

    This class encapsulates:
    - Vehicle parameters (wheelbase, timestep, etc.)
    - Scenario parameters (corridor, waypoint, horizon)
    - Disturbance parameters (base uncertainty)
    - Methods for trajectory simulation and STL formula construction

    Attributes:
        L (float): Wheelbase [m]
        dt (float): Discretization timestep [s]
        T (int): Time horizon [steps]
        x0 (np.ndarray): Initial state [x, y, θ, v]
        u_nominal (np.ndarray): Nominal control input [a, δ]
        corridor_width (float): Full corridor width [m]
        waypoint_y (tuple): (y_min, y_max) for goal lateral position [m]
        waypoint_time_window (int): Time window for waypoint reach [steps]
        sigma_y_base (float): Base lateral uncertainty [m]
        nbins (int): Number of bins for CDF discretization
    """

    # Default parameters
    DEFAULT_WHEELBASE = 2.5          # [m]
    DEFAULT_DT = 0.2                 # [s]
    DEFAULT_T_HORIZON = 25           # [steps] (5 seconds total)
    DEFAULT_X0 = np.array([0.0, 0.0, 0.0, 10.0])  # [x, y, θ, v]
    DEFAULT_U_NOMINAL = np.array([0.0, 0.001])    # [a, δ] (nearly straight)
    DEFAULT_CORRIDOR_WIDTH = 5.0     # [m]
    DEFAULT_WAYPOINT_Y = (-1.0, 1.0) # [m]
    DEFAULT_WAYPOINT_TIME_WINDOW = 5 # [steps]
    DEFAULT_SIGMA_Y_BASE = 0.15      # [m]
    DEFAULT_NBINS = 50

    def __init__(
        self,
        L=None,
        dt=None,
        T=None,
        x0=None,
        u_nominal=None,
        corridor_width=None,
        waypoint_y=None,
        waypoint_time_window=None,
        sigma_y_base=None,
        nbins=None
    ):
        """
        Initialize Bicycle model with given or default parameters.

        Args:
            L (float): Wheelbase [m]
            dt (float): Discretization timestep [s]
            T (int): Time horizon [steps]
            x0 (np.ndarray): Initial state [x, y, θ, v]
            u_nominal (np.ndarray): Nominal control input [a, δ]
            corridor_width (float): Full corridor width [m]
            waypoint_y (tuple): (y_min, y_max) for goal [m]
            waypoint_time_window (int): Time window for waypoint [steps]
            sigma_y_base (float): Base lateral uncertainty [m]
            nbins (int): CDF discretization bins
        """
        self.L = L if L is not None else self.DEFAULT_WHEELBASE
        self.dt = dt if dt is not None else self.DEFAULT_DT
        self.T = T if T is not None else self.DEFAULT_T_HORIZON
        self.x0 = x0 if x0 is not None else self.DEFAULT_X0.copy()
        self.u_nominal = u_nominal if u_nominal is not None else self.DEFAULT_U_NOMINAL.copy()
        self.corridor_width = corridor_width if corridor_width is not None else self.DEFAULT_CORRIDOR_WIDTH
        self.waypoint_y = waypoint_y if waypoint_y is not None else self.DEFAULT_WAYPOINT_Y
        self.waypoint_time_window = waypoint_time_window if waypoint_time_window is not None else self.DEFAULT_WAYPOINT_TIME_WINDOW
        self.sigma_y_base = sigma_y_base if sigma_y_base is not None else self.DEFAULT_SIGMA_Y_BASE
        self.nbins = nbins if nbins is not None else self.DEFAULT_NBINS

        # Cached nominal trajectory (computed lazily)
        self._nominal_traj = None

    @property
    def corridor_hw(self):
        """Corridor half-width [m]."""
        return self.corridor_width / 2

    @property
    def nominal_traj(self):
        """Nominal trajectory (computed lazily and cached)."""
        if self._nominal_traj is None:
            self._nominal_traj = self.simulate_nominal_trajectory()
        return self._nominal_traj

    def dynamics(self, x, u, w):
        """
        Kinematic bicycle model with additive disturbances (Euler discretization).

        Continuous-time dynamics:
            ẋ = v cos(θ)
            ẏ = v sin(θ)
            θ̇ = (v/L) tan(δ) + w_θ
            v̇ = a + w_v

        Discrete-time (Euler):
            x_{k+1} = x_k + dt · v_k cos(θ_k)
            y_{k+1} = y_k + dt · v_k sin(θ_k)
            θ_{k+1} = θ_k + dt · (v_k/L) tan(δ_k) + w_θ,k
            v_{k+1} = v_k + dt · a_k + w_v,k

        Citation: Rajamani (2012) "Vehicle Dynamics and Control", Section 2.2

        Args:
            x (np.ndarray): Current state [x, y, θ, v] in [m, m, rad, m/s]
            u (np.ndarray): Control input [a, δ] in [m/s², rad]
            w (np.ndarray): Disturbance [0, 0, w_θ, w_v] (additive noise)

        Returns:
            x_next (np.ndarray): Next state [x, y, θ, v]
        """
        x_next = np.zeros(4)

        # Position updates (kinematic)
        x_next[0] = x[0] + self.dt * x[3] * np.cos(x[2])  # x
        x_next[1] = x[1] + self.dt * x[3] * np.sin(x[2])  # y

        # Heading update (with disturbance)
        x_next[2] = x[2] + self.dt * (x[3] / self.L) * np.tan(u[1]) + w[2]  # θ

        # Velocity update (with disturbance)
        x_next[3] = x[3] + self.dt * u[0] + w[3]  # v

        return x_next

    def simulate_nominal_trajectory(self, x0=None, u=None):
        """
        Simulate nominal (disturbance-free) trajectory.

        Args:
            x0 (np.ndarray): Initial state [x, y, θ, v] (default: self.x0)
            u (np.ndarray): Constant control input [a, δ] (default: self.u_nominal)

        Returns:
            traj (np.ndarray): (4 × T+1) array of states over time
        """
        if x0 is None:
            x0 = self.x0
        if u is None:
            u = self.u_nominal

        traj = np.zeros((4, self.T + 1))
        traj[:, 0] = x0

        # Zero disturbance for nominal trajectory
        w_zero = np.zeros(4)

        for t in range(self.T):
            traj[:, t + 1] = self.dynamics(traj[:, t], u, w_zero)

        return traj

    def build_signal_gaussian(self, nominal_traj=None):
        """
        Build STL signal with Gaussian uncertainty for y-position.

        Uncertainty grows as √(t+1) for open-loop propagation.

        Args:
            nominal_traj (np.ndarray): (4 × T+1) nominal trajectory
                                       (default: self.nominal_traj)

        Returns:
            signal_y (list): List of T+1 STLRandomVariable objects for y-position
        """
        if nominal_traj is None:
            nominal_traj = self.nominal_traj

        signal_y = []

        for t in range(self.T + 1):
            # Uncertainty grows as √(t + 1) for open-loop propagation
            sqrt_t_factor = np.sqrt(t + 1)

            # Y-position uncertainty (lateral deviation)
            std_y = self.sigma_y_base * sqrt_t_factor
            mean_y = nominal_traj[1, t]

            rv_y = STLRandomVariable(
                scipy.stats.norm(loc=mean_y, scale=std_y),
                debug=False
            )
            rv_y.compute_inverse_cdf_from_cdf(
                self.nbins,
                np.interval(mean_y - 3 * std_y, mean_y + 3 * std_y)
            )
            signal_y.append(rv_y)

        return signal_y

    def build_signal_beta(self, nominal_traj=None, alpha=5, beta=2):
        """
        Build STL signal with β distribution uncertainty for y-position.

        Uncertainty grows as √(t+1) for open-loop propagation.

        Args:
            nominal_traj (np.ndarray): (4 × T+1) nominal trajectory
            alpha (float): β distribution parameter α
            beta (float): β distribution parameter β

        Returns:
            signal_y (list): List of T+1 STLRandomVariable objects for y-position
        """
        if nominal_traj is None:
            nominal_traj = self.nominal_traj

        signal_y = []

        for t in range(self.T + 1):
            sqrt_t_factor = np.sqrt(t + 1)
            std_y = self.sigma_y_base * sqrt_t_factor
            mean_y = nominal_traj[1, t]

            # Scale β distribution to have desired mean and std
            # β(α, β) has mean α/(α+β) and variance αβ/((α+β)²(α+β+1))
            beta_mean = alpha / (alpha + beta)
            beta_var = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
            beta_std = np.sqrt(beta_var)

            # Scale factor to achieve desired std_y
            scale = std_y / beta_std
            # Location to center at mean_y
            loc = mean_y - scale * beta_mean

            rv_y = STLRandomVariable(
                scipy.stats.beta(alpha, beta, loc=loc, scale=scale),
                debug=False
            )
            rv_y.compute_inverse_cdf_from_cdf(
                self.nbins,
                np.interval(loc, loc + scale)
            )
            signal_y.append(rv_y)

        return signal_y

    def create_stl_formula(self):
        """
        Construct STL formula for constrained trajectory with waypoint reach.

        Formula: φ = G[0,T](corridor) ∧ F[T-window,T](waypoint)

        Where:
        - corridor: -corridor_hw ≤ y ≤ +corridor_hw (always stay in lane)
        - waypoint: waypoint_y[0] ≤ y ≤ waypoint_y[1] (reach goal)

        Returns:
            phi (STLFormula): Combined formula
            phi_corridor (STLFormula): Corridor constraint only (always)
            phi_waypoint (STLFormula): Waypoint constraint only (eventually)
        """
        # Corridor constraint: -corridor_hw ≤ y ≤ +corridor_hw
        corridor_lower = LinearPredicate(
            np.array([1]),
            -self.corridor_hw,
            random=True
        )
        corridor_upper = LinearPredicate(
            np.array([-1]),
            -self.corridor_hw,
            random=True
        )
        corridor = corridor_lower & corridor_upper

        # Waypoint constraint: waypoint_y[0] ≤ y ≤ waypoint_y[1]
        wp_y_lower = LinearPredicate(
            np.array([1]),
            self.waypoint_y[0],
            random=True
        )
        wp_y_upper = LinearPredicate(
            np.array([-1]),
            -self.waypoint_y[1],
            random=True
        )
        waypoint = wp_y_lower & wp_y_upper

        # Temporal operators
        phi_corridor = corridor.always(0, self.T)
        wp_start = self.T - self.waypoint_time_window
        phi_waypoint = waypoint.eventually(wp_start, self.T)

        # Combined formula
        phi = phi_corridor & phi_waypoint

        return phi, phi_corridor, phi_waypoint

    def create_corridor_formula(self):
        """
        Create corridor-only STL formula: φ = G[0,T](corridor)

        Returns:
            phi_corridor (STLFormula): Always stay in corridor
        """
        corridor_lower = LinearPredicate(
            np.array([1]),
            -self.corridor_hw,
            random=True
        )
        corridor_upper = LinearPredicate(
            np.array([-1]),
            -self.corridor_hw,
            random=True
        )
        corridor = corridor_lower & corridor_upper
        phi_corridor = corridor.always(0, self.T)
        return phi_corridor

    def __repr__(self):
        return (
            f"Bicycle(L={self.L}, dt={self.dt}, T={self.T}, "
            f"corridor_width={self.corridor_width}, "
            f"waypoint_y={self.waypoint_y}, "
            f"sigma_y_base={self.sigma_y_base})"
        )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def construct_ar1_correlation(n_steps, rho):
    """
    Construct AR(1) (autoregressive order 1) temporal correlation matrix.

    The AR(1) process models temporal dependence where correlation decays
    exponentially with time lag:
        Σ_ij = ρ^|i - j|

    Citation: Box & Jenkins (1976) "Time Series Analysis", Section 3.2

    Args:
        n_steps (int): Number of timesteps (correlation matrix dimension)
        rho (float): Temporal correlation coefficient ρ ∈ [0, 1]

    Returns:
        Sigma (np.ndarray): (n_steps × n_steps) correlation matrix
    """
    Sigma = np.zeros((n_steps, n_steps))
    for i in range(n_steps):
        for j in range(n_steps):
            Sigma[i, j] = rho ** abs(i - j)
    return Sigma


# ============================================================================
# BACKWARD COMPATIBILITY: Module-level constants and functions
# ============================================================================

# Default parameters (for backward compatibility with existing imports)
L_WHEELBASE = Bicycle.DEFAULT_WHEELBASE
DT = Bicycle.DEFAULT_DT
T_HORIZON = Bicycle.DEFAULT_T_HORIZON
X0 = Bicycle.DEFAULT_X0.copy()
U_NOMINAL = Bicycle.DEFAULT_U_NOMINAL.copy()
CORRIDOR_WIDTH = Bicycle.DEFAULT_CORRIDOR_WIDTH
WAYPOINT_Y = Bicycle.DEFAULT_WAYPOINT_Y
WAYPOINT_TIME_WINDOW = Bicycle.DEFAULT_WAYPOINT_TIME_WINDOW
SIGMA_Y_BASE = Bicycle.DEFAULT_SIGMA_Y_BASE
NBINS = Bicycle.DEFAULT_NBINS


def simulate_nominal_trajectory(x0, u, T, dt=DT, L=L_WHEELBASE):
    """
    Simulate nominal trajectory (backward-compatible function).

    Args:
        x0 (np.ndarray): Initial state [x, y, θ, v]
        u (np.ndarray): Constant control input [a, δ]
        T (int): Time horizon [steps]
        dt (float): Timestep [s]
        L (float): Wheelbase [m]

    Returns:
        traj (np.ndarray): (4 × T+1) array of states over time
    """
    bike = Bicycle(L=L, dt=dt, T=T, x0=x0, u_nominal=u)
    return bike.simulate_nominal_trajectory()


def create_bicycle_stl_formula(T, corridor_hw, waypoint_y, wp_window):
    """
    Construct STL formula (backward-compatible function).

    Args:
        T (int): Time horizon [steps]
        corridor_hw (float): Corridor half-width [m]
        waypoint_y (list): [y_min, y_max] for goal y-coordinate [m]
        wp_window (int): Time window for waypoint reach [steps]

    Returns:
        phi (STLFormula): Combined formula
        corridor (STLFormula): Corridor predicate (not temporal)
        waypoint (STLFormula): Waypoint predicate (not temporal)
    """
    # Create corridor predicate
    corridor_lower = LinearPredicate(
        np.array([1]),
        -corridor_hw,
        random=True
    )
    corridor_upper = LinearPredicate(
        np.array([-1]),
        -corridor_hw,
        random=True
    )
    corridor = corridor_lower & corridor_upper

    # Create waypoint predicate
    wp_y_lower = LinearPredicate(
        np.array([1]),
        waypoint_y[0],
        random=True
    )
    wp_y_upper = LinearPredicate(
        np.array([-1]),
        -waypoint_y[1],
        random=True
    )
    waypoint = wp_y_lower & wp_y_upper

    # Temporal operators
    phi_corridor_always = corridor.always(0, T)
    phi_waypoint_eventually = waypoint.eventually(T - wp_window, T)

    # Combined formula
    phi = phi_corridor_always & phi_waypoint_eventually

    return phi, corridor, waypoint


def create_bicycle_stl_formula_2copula(T, corridor_hw, waypoint_y, wp_window):
    """
    Construct STL formula with structure suitable for pairwise 2-copula evaluation.

    This uses an alternative structure that separates temporal operators from
    predicate conjunction. Instead of:
        always[0,T]((y >= -hw) AND (y <= hw))
    We use:
        always[0,T](y >= -hw) AND always[0,T](y <= hw)

    This structure allows the Gaussian 2-copula to properly model temporal
    correlation without incorrectly applying it to predicates at the same timestep.

    Args:
        T (int): Time horizon [steps]
        corridor_hw (float): Corridor half-width [m]
        waypoint_y (list): [y_min, y_max] for goal y-coordinate [m]
        wp_window (int): Time window for waypoint reach [steps]

    Returns:
        phi (STLFormula): Combined formula with 2-copula compatible structure
    """
    # Create corridor predicates
    corridor_lower = LinearPredicate(
        np.array([1]),
        -corridor_hw,
        random=True
    )
    corridor_upper = LinearPredicate(
        np.array([-1]),
        -corridor_hw,
        random=True
    )

    # Create waypoint predicates
    wp_y_lower = LinearPredicate(
        np.array([1]),
        waypoint_y[0],
        random=True
    )
    wp_y_upper = LinearPredicate(
        np.array([-1]),
        -waypoint_y[1],
        random=True
    )

    # Apply temporal operators BEFORE conjunction (key difference from original)
    phi_corridor_lower = corridor_lower.always(0, T)
    phi_corridor_upper = corridor_upper.always(0, T)
    phi_corridor = phi_corridor_lower & phi_corridor_upper

    wp_start = T - wp_window
    phi_wp_lower = wp_y_lower.eventually(wp_start, T)
    phi_wp_upper = wp_y_upper.eventually(wp_start, T)
    phi_waypoint = phi_wp_lower & phi_wp_upper

    # Combined formula
    phi = phi_corridor & phi_waypoint

    return phi


def build_signal_with_uncertainty(nominal_traj, T, nbins=NBINS):
    """
    Build STL signal with Gaussian uncertainty (backward-compatible function).

    Args:
        nominal_traj (np.ndarray): (4 × T+1) nominal trajectory
        T (int): Time horizon [steps]
        nbins (int): CDF discretization bins

    Returns:
        signal_y (list): List of T+1 STLRandomVariable objects for y-position
    """
    bike = Bicycle(T=T, nbins=nbins)
    return bike.build_signal_gaussian(nominal_traj)
