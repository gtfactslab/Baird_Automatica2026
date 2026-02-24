"""
Generate setup diagram for bicycle scenario (paper-ready, PDF output)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Import from Bicycle module
from Bicycle import Bicycle, CORRIDOR_WIDTH, WAYPOINT_Y

# LaTeX rendering for publication-quality plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['font.size'] = 10

np.random.seed(42)


# ============================================================================
# BICYCLE REPRESENTATION
# ============================================================================

def draw_bicycle(ax, x, y, theta, scale=1.0, color='blue'):
    """
    Draw a simplified bicycle using rectangles.

    Args:
        ax: matplotlib axis
        x, y: position
        theta: heading angle [rad]
        scale: scaling factor for size
        color: color of bicycle
    """
    # Bicycle dimensions (scaled)
    length = 2.5 * scale  # wheelbase
    width = 0.8 * scale   # body width
    wheel_radius = 0.3 * scale
    wheel_width = 0.15 * scale

    # Rotation matrix
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])

    # Body rectangle (centered at (x, y))
    body_local = np.array([
        [-length/2, -width/2],
        [length/2, -width/2],
        [length/2, width/2],
        [-length/2, width/2],
        [-length/2, -width/2]
    ])
    body_global = (R @ body_local.T).T + np.array([x, y])
    ax.plot(body_global[:, 0], body_global[:, 1], color=color, linewidth=2)
    ax.fill(body_global[:, 0], body_global[:, 1], color=color, alpha=0.3)

    # Rear wheel (left side)
    wheel_rear_local = np.array([-length/2, 0])
    wheel_rear_global = R @ wheel_rear_local + np.array([x, y])
    wheel_rear = patches.Rectangle(
        (wheel_rear_global[0] - wheel_width/2, wheel_rear_global[1] - wheel_radius),
        wheel_width, 2*wheel_radius,
        angle=np.degrees(theta),
        color='black',
        fill=True
    )
    ax.add_patch(wheel_rear)

    # Front wheel (right side)
    wheel_front_local = np.array([length/2, 0])
    wheel_front_global = R @ wheel_front_local + np.array([x, y])
    wheel_front = patches.Rectangle(
        (wheel_front_global[0] - wheel_width/2, wheel_front_global[1] - wheel_radius),
        wheel_width, 2*wheel_radius,
        angle=np.degrees(theta),
        color='black',
        fill=True
    )
    ax.add_patch(wheel_front)


# ============================================================================
# SETUP DIAGRAM
# ============================================================================

def plot_setup_diagram_pdf(traj, corridor_hw, waypoint_y):
    """
    Plot bicycle scenario setup diagram (paper-ready, PDF output).

    - Compact height for paper
    - Bicycle representation using rectangles
    - Clean, publication-quality formatting
    """
    # Figure size: 5.5x2 inches
    fig, ax = plt.subplots(figsize=(5.5, 2))

    # Nominal trajectory
    ax.plot(traj[0, :], traj[1, :], 'b-', linewidth=1.5, label='Nominal Trajectory',
            zorder=3)

    # Corridor constraints
    ax.axhline(y=corridor_hw, color='k', linestyle='--', linewidth=1.2,
               label='Corridor Bounds', zorder=2)
    ax.axhline(y=-corridor_hw, color='k', linestyle='--', linewidth=1.2, zorder=2)

    # Fill corridor region
    ax.fill_between(
        [traj[0, 0] - 5, traj[0, -1] + 10],
        [-corridor_hw, -corridor_hw],
        [corridor_hw, corridor_hw],
        alpha=0.15, color='green', label='Safe Corridor', zorder=1
    )

    # Waypoint region
    wp_x_min = traj[0, -1] - 3
    wp_x_max = traj[0, -1] + 10
    ax.fill_between(
        [wp_x_min, wp_x_max],
        [waypoint_y[0], waypoint_y[0]],
        [waypoint_y[1], waypoint_y[1]],
        alpha=0.25, color='red', label='Goal Region', zorder=1
    )

    # Waypoint bounds
    ax.axhline(y=waypoint_y[0], color='red', linestyle=':', linewidth=1.2,
               xmin=0.75, xmax=1.0, zorder=2)
    ax.axhline(y=waypoint_y[1], color='red', linestyle=':', linewidth=1.2,
               xmin=0.75, xmax=1.0, zorder=2)

    # Draw bicycle at start position
    draw_bicycle(ax, traj[0, 0], traj[1, 0], traj[2, 0], scale=0.8, color='green')

    # Draw bicycle at end position
    draw_bicycle(ax, traj[0, -1], traj[1, -1], traj[2, -1], scale=0.8, color='red')

    # Add whitespace on sides (10 meters on each side)
    x_margin = 10
    ax.set_xlim(traj[0, 0] - x_margin, traj[0, -1] + x_margin)

    # Set y-axis limits: -3.5 to +3.5m
    ax.set_ylim(-3.5, 3.5)

    # Labels and formatting
    ax.set_xlabel(r'$x$ [m]', fontsize=11)
    ax.set_ylabel(r'$y$ [m]', fontsize=11)
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # Tight layout for paper
    plt.tight_layout()

    # Save as PDF
    plt.savefig('bicycle_setup_diagram.pdf', dpi=300, bbox_inches='tight',
                backend='pdf')
    print("Saved: bicycle_setup_diagram.pdf")

    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("Generating bicycle setup diagram for paper...")

    # Create bicycle model and generate nominal trajectory
    bike = Bicycle()
    traj = bike.nominal_traj

    # Plot setup diagram
    plot_setup_diagram_pdf(traj, bike.corridor_hw, WAYPOINT_Y)

    print("Done!")


if __name__ == "__main__":
    main()
